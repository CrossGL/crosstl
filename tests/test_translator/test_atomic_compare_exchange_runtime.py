"""Compare-exchange preserves storage, Boolean results and expected writeback."""

import os
import shutil
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_atomic_load_runtime import GUARD, _report
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_ATOMIC_COMPARE_EXCHANGE_RUNTIME"
CASES = (
    "scalar",
    "member",
    "pointer",
    "helper",
    "conditional",
    "aggregate",
    "workgroup",
    "contention",
    "collision",
)


def _source(kind, case):
    declarations = (
        f"struct Counter {{ {kind} before; atomic<{kind}> value; {kind} after; }};"
    )
    scalar = case in {"scalar", "pointer", "contention"}
    storage = f"atomic_{kind}" if scalar else "Counter"
    member = "" if scalar else ".value"
    setup = ""
    target = f"&values[position++]{member}"
    retry_target = f"&values[tid + 2u]{member}"
    call = f"atomic_compare_exchange_weak_explicit({target}, &expected[slot++], desired++, memory_order_relaxed, memory_order_relaxed)"
    retry = f"atomic_compare_exchange_weak_explicit({retry_target}, &expected[0], {kind}(17), memory_order_relaxed, memory_order_relaxed)"
    if case == "pointer":
        setup = f"device atomic<{kind}>* pointer = values + position++;"
        call = call.replace(target, "pointer")
    if case in {"helper", "aggregate"}:
        parameter = "device Counter* source"
        helper_target = "&source[index].value"
        actual = "values"
        if case == "aggregate":
            declarations += "struct View { device Counter* source; };"
            parameter = "const thread View& view"
            helper_target = "&view.source[index].value"
            setup = "View view{values}; view.source += 2; position = tid;"
            actual = "view"
        declarations += f"""bool exchange({parameter}, uint index, thread {kind}* expected, {kind} desired) {{
            return metal::atomic_compare_exchange_weak_explicit({helper_target}, expected, desired,
                metal::memory_order_relaxed, metal::memory_order_relaxed);
        }}"""
        call = f"exchange({actual}, position++, &expected[0], desired++)"
        retry = f"exchange({actual}, {'tid' if case == 'aggregate' else 'tid + 2u'}, &expected[0], {kind}(17))"
    if case == "conditional":
        call = f"tid == 1u ? false : {call}"
    body = f"""{kind} expected[3] = {{{kind}(0), {kind}({GUARD}), {kind}({GUARD})}};
        {kind} desired = {kind}(11);
        uint slot = 0u;
        bool first = {call};
        {'slot += 1u;' if case in {'helper', 'aggregate'} else ''}
        results[tid * 4u] = {kind}(first);
        results[tid * 4u + 1u] = expected[0];
        bool matched = false;
        while (!matched) {{ matched = {retry}; }}
        results[tid * 4u + 2u] = {kind}(matched);
        results[tid * 4u + 3u] = expected[0];
        counts[tid * 5u] = position;
        counts[tid * 5u + 1u] = slot;
        counts[tid * 5u + 2u] = uint(desired);
        counts[tid * 5u + 3u] = uint(expected[1]);
        counts[tid * 5u + 4u] = uint(expected[2]);"""
    if case in {"workgroup", "contention"}:
        target = "&counter[0]" if case == "workgroup" else "&values[2]"
        if case == "workgroup":
            setup = f"""threadgroup atomic_{kind} counter[1];
            if (tid == 0u) {{ atomic_store_explicit(&counter[0], {kind}(0), memory_order_relaxed); }}
            threadgroup_barrier(mem_flags::mem_threadgroup);"""
        body = f"""{kind} expected = atomic_load_explicit({target}, memory_order_relaxed);
        while (!atomic_compare_exchange_weak_explicit({target}, &expected, expected + {kind}(1),
            memory_order_relaxed, memory_order_relaxed)) {{}}
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        results[tid] = atomic_load_explicit({target}, memory_order_relaxed);
        counts[tid] = 1u;"""
    source = f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void exchange_values(device {storage}* values [[buffer(0)]],
    device {kind}* results [[buffer(1)]], device uint* counts [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {{
    {'uint position = tid + 2u;' if case not in {'workgroup', 'contention'} else ''}
    {setup}
    {body}
}}
"""
    if case == "collision":
        source = source.replace("* values [[", "* crossgl_atomic_observed [[").replace(
            "&values[", "&crossgl_atomic_observed["
        )
    return source


def _request(root, target, kind, case):
    concurrent = case in {"workgroup", "contention"}
    count = 32 if concurrent else 3
    width = count if concurrent else 1
    original, descriptor, package = _package(
        root,
        target,
        kind,
        (width, 1, 1),
        source=_source(kind, case),
        software_subgroups=False,
        index_range_assertions=(
            (
                {
                    "source": "products.metal",
                    "expression": "reference.offset + index",
                    "minimum": 2,
                    "maximum": 4,
                },
            )
            if target == "opengl" and case == "aggregate"
            else ()
        ),
    )
    values = (
        [-7, -2147483648, 2147483647] if kind == "int" else [7, 2147483648, 4294967295]
    )
    stride = 1 if case in {"scalar", "pointer", "contention"} else 3
    initial = [GUARD] * (7 * stride)
    for index, value in enumerate(values):
        initial[(index + 2) * stride + (stride != 1)] = value
    final = initial.copy()
    result = [GUARD] * ((count if concurrent else count * 4) + 3)
    counts = [GUARD] * ((count if concurrent else count * 5) + 3)
    if concurrent:
        result[:count] = [count] * count
        counts[:count] = [1] * count
        if case == "contention":
            initial[2], final[2] = 0, count
    else:
        for index, value in enumerate(values):
            active = not (case == "conditional" and index == 1)
            result[index * 4 : index * 4 + 4] = [0, value if active else 0, 1, value]
            counts[index * 5 : index * 5 + 5] = [
                index + (0 if case == "aggregate" else 2) + int(active),
                int(active),
                11 + int(active),
                GUARD,
                GUARD,
            ]
            final[(index + 2) * stride + (stride != 1)] = 17
    dtype = f"{kind}32"
    inputs = {
        "values": {
            "dtype": dtype,
            "shape": [7] if stride == 1 else [7, stride],
            "values": initial,
        },
        "results": {
            "dtype": dtype,
            "shape": [len(result)],
            "values": [GUARD] * len(result),
        },
        "counts": {
            "dtype": "uint32",
            "shape": [len(counts)],
            "values": [GUARD] * len(counts),
        },
    }
    expected = _bound_values(
        descriptor,
        {
            "crossgl_atomic_observed" if case == "collision" else "values": {
                **inputs["values"],
                "values": final,
            },
            "results": {**inputs["results"], "values": result},
            "counts": {**inputs["counts"], "values": counts},
        },
    )
    if case == "collision":
        inputs["crossgl_atomic_observed"] = inputs.pop("values")
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
@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_atomic_compare_exchange_translates(tmp_path, target, kind, case):
    _, request, _ = _request(tmp_path, target, kind, case)
    generated = request.artifact_path.read_text()
    assert {
        "metal": "atomic_compare_exchange_weak_explicit",
        "directx": "InterlockedCompareExchange",
        "opengl": "atomicCompSwap",
    }[target] in generated
    assert "atomicCompareExchangeWeak(" not in generated
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if shutil.which(tool):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_atomic_compare_exchange_executes_natively(tmp_path, kind, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required compare-exchange execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, kind, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="exchange_values",
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "order",
    ("memory_order_acquire", "memory_order_seq_cst", "other::memory_order_relaxed"),
)
def test_atomic_compare_exchange_order_is_not_weakened(tmp_path, target, order):
    report = _report(
        tmp_path,
        _source("int", "scalar").replace("memory_order_relaxed", order),
        target,
    )
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metal-atomic-compare-exchange-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "change",
    ("readonly", "mismatched", "private", "arity", "shadowed-order", "name-collision"),
)
def test_atomic_compare_exchange_unsupported_contract_fails_closed(
    tmp_path, target, change
):
    source = _source("int", "scalar")
    if change == "readonly":
        source = source.replace(
            "device atomic_int* values", "const device atomic_int* values"
        )
    elif change == "mismatched":
        source = source.replace("int expected[3]", "uint expected[3]")
    elif change == "private":
        source = source.replace("&values[position++]", "&expected[0]")
    elif change == "arity":
        source = source.replace(
            ", memory_order_relaxed, memory_order_relaxed)", ", memory_order_relaxed)"
        )
    elif change == "shadowed-order":
        source = source.replace(
            "uint position =", "int memory_order_relaxed = 0; uint position ="
        )
    else:
        source = source.replace(
            "kernel void",
            "bool atomicCompareExchangeWeak(int value) { return value != 0; } kernel void",
        )
    report = _report(tmp_path, source, target)
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metal-atomic-compare-exchange-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_compare_helper_rejects_side_effecting_private_pointer_view(tmp_path, target):
    source = _source("int", "helper").replace(
        "&expected[0], desired++", "&expected[slot++], desired++"
    )
    report = _report(tmp_path, source, target)
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == f"project.translate.{target}-private-pointer-unsupported"
        for item in report["diagnostics"]
    )

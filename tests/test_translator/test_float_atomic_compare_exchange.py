"""Floating compare-exchange preserves payloads and expected-value writeback."""

import json
import os
import re
import shutil
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from tests.test_translator.metal_float_atomic_reference import (
    shared_float_reference_flags,
)
from tests.test_translator.test_atomic_load_runtime import GUARD, _report
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_float_storage_encoding import WORDS
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_FLOAT_ATOMIC_COMPARE_EXCHANGE"
CASES = (
    "scalar",
    "member",
    "pointer",
    "helper",
    "aggregate",
    "conditional",
    "workgroup",
    "collision",
)


def _source(case):
    scalar = case in {"scalar", "pointer"}
    declarations = "struct Counter { float before; atomic_float value; float after; };"
    storage, member = ("atomic_float", "") if scalar else ("Counter", ".value")
    setup = ""
    first_target = f"&values[position++]{member}"
    target = f"&values[tid + 2u]{member}"
    expected = "&expected[slot++]"
    desired = "as_type<float>(incoming[value_position++])"
    if case == "pointer":
        setup = "device atomic_float* base = values + position++;"
        first_target = target = "base"
    if case in {"workgroup", "collision"}:
        name = "__crossgl_cas_observed" if case == "collision" else "shared_values"
        setup = f"""threadgroup atomic_float {name}[{len(WORDS)}];
            atomic_store_explicit(&{name}[tid], atomic_load_explicit({target}, memory_order_relaxed), memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            position = tid;"""
        first_target, target = f"&{name}[position++]", f"&{name}[tid]"
    first = f"atomic_compare_exchange_weak_explicit({first_target}, {expected}, {desired}, memory_order_relaxed, memory_order_relaxed)"
    retry = f"atomic_compare_exchange_weak_explicit({target}, &expected[0], as_type<float>(incoming[tid * 2u + 1u]), memory_order_relaxed, memory_order_relaxed)"
    if case in {"helper", "aggregate"}:
        parameter, actual, helper_target = (
            "device Counter* source",
            "values",
            "&source[index].value",
        )
        if case == "aggregate":
            declarations += "struct View { device Counter* source; };"
            parameter, actual, helper_target = (
                "const thread View& view",
                "view",
                "&view.source[index].value",
            )
            setup = "View view{values}; view.source += 2; position = tid;"
        declarations += f"""bool compare({parameter}, uint index, thread float* expected, float desired) {{
            return atomic_compare_exchange_weak_explicit({helper_target}, expected, desired, memory_order_relaxed, memory_order_relaxed);
        }}"""
        first = f"compare({actual}, position++, &expected[0], {desired})"
        retry = f"compare({actual}, tid {'+ 2u' if case == 'helper' else ''}, &expected[0], as_type<float>(incoming[tid * 2u + 1u]))"
    body = f"""bool first = {first};
        {'++slot;' if case in {'helper', 'aggregate'} else ''}
        results[tid * 6u] = uint(first);
        results[tid * 6u + 1u] = as_type<uint>(expected[0]);
        bool success = false;
        for (uint attempt = 0u; attempt < 8u && !success; ++attempt) {{
            success = {retry};
        }}
        results[tid * 6u + 2u] = uint(success);
        results[tid * 6u + 3u] = as_type<uint>(expected[0]);
        results[tid * 6u + 4u] = as_type<uint>(atomic_load_explicit({target}, memory_order_relaxed));
        results[tid * 6u + 5u] = as_type<uint>(expected[1]);"""
    if case == "conditional":
        body = f"if ((tid % 2u) == 0u) {{ {body} }}"
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void compare_values(device {storage}* values [[buffer(0)]],
                          device uint* incoming [[buffer(1)]],
                          device uint* results [[buffer(2)]],
                          device uint* counts [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {{
    uint position = tid + 2u;
    uint slot = 0u;
    uint value_position = tid * 2u + 1u;
    float expected[2] = {{as_type<float>(incoming[tid * 2u]), as_type<float>({GUARD}u)}};
    {setup}
    {body}
    counts[tid * 3u] = position;
    counts[tid * 3u + 1u] = slot;
    counts[tid * 3u + 2u] = value_position;
}}
"""


def _request(root, target, case):
    count = len(WORDS)
    shared = case in {"workgroup", "collision"}
    source, descriptor, package = _package(
        root,
        target,
        "float",
        (count if shared else 1, 1, 1),
        source=_source(case),
        software_subgroups=False,
        index_range_assertions=(
            (
                {
                    "source": "products.metal",
                    "expression": "reference.offset + index",
                    "minimum": 2,
                    "maximum": count + 1,
                },
            )
            if target == "opengl" and case == "aggregate"
            else ()
        ),
    )
    stride = 1 if case in {"scalar", "pointer"} else 3
    initial = [GUARD] * ((count + 4) * stride)
    final = initial.copy()
    incoming = []
    results, counts = [GUARD] * (count * 6 + 3), [GUARD] * (count * 3 + 3)
    for index, bits in enumerate(WORDS):
        desired = WORDS[count - index - 1]
        incoming.extend((bits ^ 0x80000000, desired))
        offset = (index + 2) * stride + (stride != 1)
        initial[offset] = final[offset] = bits
        active = case != "conditional" or index % 2 == 0
        if active:
            results[index * 6 : index * 6 + 6] = [0, bits, 1, bits, desired, GUARD]
            if not shared:
                final[offset] = desired
        counts[index * 3 : index * 3 + 3] = [
            index + (0 if shared or case == "aggregate" else 2) + int(active),
            int(active),
            index * 2 + 1 + int(active),
        ]

    def typed(values, floating=False):
        return {
            "dtype": "float32" if floating and target != "opengl" else "uint32",
            "shape": [count + 4, stride] if floating and stride > 1 else [len(values)],
            "values": values,
            **({"encoding": FLOAT32_BITS} if floating and target != "opengl" else {}),
        }

    inputs = {
        "values": typed(initial, True),
        "incoming": typed(incoming),
        "results": typed([GUARD] * len(results)),
        "counts": typed([GUARD] * len(counts)),
    }
    expected = _bound_values(
        descriptor,
        {
            "values": typed(final, True),
            "incoming": typed(incoming),
            "results": typed(results),
            "counts": typed(counts),
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {
            "workgroupCount": [1 if shared else count, 1, 1],
            "workgroupSize": [count if shared else 1, 1, 1],
        },
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    (root / "workload.json").write_text(
        json.dumps({"case": case, "inputs": inputs, "outputs": expected}, indent=2),
        encoding="utf-8",
    )
    return source, request, expected


@pytest.mark.parametrize("target", ("metal", "directx"))
@pytest.mark.parametrize("case", CASES)
def test_float_compare_exchange_translates_and_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    assert "atomicCompareExchangeWeak(" not in generated
    if target == "directx":
        assert "InterlockedCompareExchangeFloatBitwise" in generated
        assert re.search(r"asuint\(\w+\) == asuint\(\w+\)", generated)
    if shutil.which("xcrun" if target == "metal" else "dxc"):
        _, module = _compile(
            generated,
            target,
            tmp_path,
            metal_compile_flags=("-std=metal4.0",) if target == "metal" else (),
        )
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_float_compare_exchange_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native floating compare-exchange")
    assert sys.platform in {"darwin", "win32"}
    target = "metal" if sys.platform == "darwin" else "directx"
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="compare_values",
        metal_compile_flags=(
            shared_float_reference_flags(tmp_path)
            if case in {"workgroup", "collision"}
            else ()
        ),
    )


@pytest.mark.parametrize("case", CASES)
def test_float_compare_exchange_opengl_preserves_index_contract(tmp_path, case):
    report = _report(tmp_path, _source(case), "opengl")
    if case == "aggregate":
        assert report["summary"]["failedCount"] == 1
        assert all(
            not (tmp_path / artifact["path"]).exists()
            for artifact in report["artifacts"]
        )
        assert [item["code"] for item in report["diagnostics"]] == [
            "project.translate.opengl-index-type-unsupported"
        ]
    else:
        assert report["summary"]["failedCount"] == 0
        generated = (tmp_path / report["artifacts"][0]["path"]).read_text()
        assert "atomicCompSwap(" in generated
        assert "crossgl_pending_float_atomic" not in generated


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "change", ("readonly", "expected-type", "expected-readonly", "order")
)
def test_float_compare_exchange_invalid_contract_is_not_published(
    tmp_path, target, change
):
    source = _source("scalar")
    if change == "readonly":
        source = source.replace(
            "device atomic_float* values", "const device atomic_float* values"
        )
    elif change == "expected-type":
        source = source.replace("float expected[2]", "uint expected[2]")
    elif change == "expected-readonly":
        source = source.replace("float expected[2]", "const float expected[2]")
    else:
        source = source.replace(
            "memory_order_relaxed, memory_order_relaxed)",
            "memory_order_relaxed, memory_order_acquire)",
        )
    report = _report(tmp_path, source, target)
    assert report["summary"]["translatedCount"] == 0
    assert all(
        not (tmp_path / artifact["path"]).exists() for artifact in report["artifacts"]
    )
    assert any(
        item["code"] == "project.translate.metal-atomic-compare-exchange-unsupported"
        for item in report["diagnostics"]
    )

"""Floating atomic memory preserves stored bits and resource identity."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from tests.test_translator.metal_float_atomic_reference import (
    shared_float_reference_flags,
)
from tests.test_translator.test_atomic_load_runtime import GUARD, _report
from tests.test_translator.test_atomic_resource_runtime import (
    _source as _resource_source,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_float_storage_encoding import WORDS
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_FLOAT_ATOMIC_MEMORY"
SHARED_CASES = (
    "workgroup",
    "collision",
    "nested-conditional",
    "nested-return",
    "nested-result",
)
CASES = (
    "scalar",
    "member",
    "pointer",
    "helper",
    "aggregate",
    "conditional",
    "loop",
    *SHARED_CASES,
)


def _source(operation, case):
    scalar = case in {"scalar", "pointer"}
    declarations = "struct Counter { float before; atomic_float value; float after; };"
    storage, member = ("atomic_float", "") if scalar else ("Counter", ".value")
    setup = ""
    target = f"&values[position++]{member}"
    retry_target = f"&values[tid + 2u]{member}"
    incoming = "as_type<float>(incoming[value_position++])"
    load = f"atomic_load_explicit({target}, memory_order_relaxed)"
    store = f"atomic_store_explicit({target}, {incoming}, memory_order_relaxed);"
    read = f"atomic_load_explicit({retry_target}, memory_order_relaxed)"
    if case == "pointer":
        setup = "device atomic_float* base = values + position++;"
        load = "atomic_load_explicit(base, memory_order_relaxed)"
        store = f"atomic_store_explicit(base, {incoming}, memory_order_relaxed);"
    elif case in {"helper", "aggregate"}:
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
        declarations += f"""float load({parameter}, uint index) {{
            return atomic_load_explicit({helper_target}, memory_order_relaxed);
        }}
        void store({parameter}, uint index, float value) {{
            atomic_store_explicit({helper_target}, value, memory_order_relaxed);
        }}"""
        load = f"load({actual}, position++)"
        store = f"store({actual}, position++, {incoming});"
    body = (
        f"results[tid] = as_type<uint>({load});"
        if operation == "load"
        else f"{store} results[tid] = as_type<uint>({read});"
    )
    if case == "conditional":
        body = f"if ((tid % 2u) == 0u) {{ {body} }}"
    elif case == "loop":
        body = f"for (uint pass = 0u; pass < 2u; ++pass) {{ {body} --position; {'--value_position;' if operation == 'store' else ''} }}"
    elif case in SHARED_CASES:
        # Every invocation initializes a distinct cell before its neighbour reads it.
        name = "original" if case == "collision" else "shared_values"
        initialize = (
            f"atomic_store_explicit(&{name}[tid], {read}, memory_order_relaxed);"
        )
        if case == "nested-conditional":
            initialize = f"atomic_store_explicit(&{name}[tid], (tid % 2u) == 0u ? {load} : {load}, memory_order_relaxed);"
        elif case == "nested-return":
            declarations += """void initialize(threadgroup atomic_float* shared,
                                               device Counter* input, uint index) {
                return atomic_store_explicit(&shared[index],
                    atomic_load_explicit(&input[index + 2u].value, memory_order_relaxed),
                    memory_order_relaxed);
            }"""
            initialize = f"initialize({name}, values, tid);"
        elif case == "nested-result":
            initialize = f"""float observed = atomic_exchange_explicit({target}, {read}, memory_order_relaxed);
                atomic_store_explicit(&{name}[tid], observed, memory_order_relaxed);"""
        setup = f"""threadgroup atomic_float {name}[{len(WORDS)}];
            {initialize}
            threadgroup_barrier(mem_flags::mem_threadgroup);"""
        body = (
            (
                f"atomic_store_explicit(&{name}[tid], {incoming}, memory_order_relaxed);"
                "threadgroup_barrier(mem_flags::mem_threadgroup);"
                if operation == "store"
                else ""
            )
            + f"results[tid] = as_type<uint>(atomic_load_explicit(&{name}[(tid + 1u) % {len(WORDS)}u], memory_order_relaxed));"
        )
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void atomic_memory(device {storage}* values [[buffer(0)]],
                          device uint* incoming [[buffer(1)]],
                          device uint* results [[buffer(2)]],
                          device uint* counts [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {{
    uint position = tid + 2u;
    uint value_position = tid;
    {setup}
    {body}
    counts[2u * tid] = position;
    counts[2u * tid + 1u] = value_position;
}}
"""


def _request(root, target, operation, case):
    count = len(WORDS)
    shared = case in SHARED_CASES
    source, descriptor, package = _package(
        root,
        target,
        "float",
        (count if shared else 1, 1, 1),
        source=_source(operation, case),
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
        workgroup_access_assertions=(
            (
                {
                    "source": "products.metal",
                    "entry_point": "atomic_memory",
                    "function": "initialize",
                    "parameter": "shared",
                    "minimum": 0,
                    "maximum": count - 1,
                },
            )
            if target == "opengl" and case == "nested-return"
            else ()
        ),
    )
    stride = 1 if case in {"scalar", "pointer"} else 3
    initial = [GUARD] * ((count + 4) * stride)
    final = initial.copy()
    updated = list(reversed(WORDS))
    output, counts = [GUARD] * (count + 3), [GUARD] * (2 * count + 3)
    for i, bits in enumerate(WORDS):
        offset = (i + 2) * stride + (stride != 1)
        initial[offset] = final[offset] = bits
        active = case != "conditional" or i % 2 == 0
        if active:
            output[i] = (updated if operation == "store" else WORDS)[
                (i + 1) % count if shared else i
            ]
            if operation == "store" and not shared:
                final[offset] = updated[i]
        counts[2 * i] = (
            i
            + (0 if case == "aggregate" else 2)
            + int(active and not shared and case != "loop")
            + int(case in {"nested-conditional", "nested-result"})
        )
        counts[2 * i + 1] = i + int(active and operation == "store" and case != "loop")

    def typed(values, *, floating=False, shape=None):
        floating = floating and target != "opengl"
        return {
            "dtype": "float32" if floating else "uint32",
            "shape": shape or [len(values)],
            "values": values,
            **({"encoding": FLOAT32_BITS} if floating else {}),
        }

    shape = [count + 4] if stride == 1 else [count + 4, stride]
    inputs = {
        "values": typed(initial, floating=True, shape=shape),
        "incoming": typed(updated),
        "results": typed([GUARD] * len(output)),
        "counts": typed([GUARD] * len(counts)),
    }
    expected = _bound_values(
        descriptor,
        {
            "values": typed(final, floating=True, shape=shape),
            "incoming": typed(updated),
            "results": typed(output),
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
        json.dumps({"operation": operation, "case": case, "count": count}, indent=2),
        encoding="utf-8",
    )
    return source, request, expected


@pytest.mark.parametrize("target", ("metal", "directx"))
@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize("case", CASES)
def test_float_atomic_memory_translates_and_compiles(tmp_path, target, operation, case):
    _, request, _ = _request(tmp_path, target, operation, case)
    generated = request.artifact_path.read_text()
    assert "atomicLoad(" not in generated and "atomicStore(" not in generated
    assert (
        "atomic_load_explicit"
        if target == "metal"
        else "InterlockedCompareExchangeFloatBitwise"
    ) in generated
    if shutil.which("xcrun" if target == "metal" else "dxc"):
        _, module = _compile(
            generated,
            target,
            tmp_path,
            metal_compile_flags=("-std=metal4.0",) if target == "metal" else (),
        )
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize("case", CASES)
def test_float_atomic_memory_executes_natively(tmp_path, operation, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native floating atomic memory execution")
    assert sys.platform in {
        "darwin",
        "win32",
    }, "Float atomic memory requires Metal or DirectX"
    target = "metal" if sys.platform == "darwin" else "directx"
    source, request, expected = _request(tmp_path, target, operation, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="atomic_memory",
        metal_compile_flags=(
            shared_float_reference_flags(tmp_path) if case in SHARED_CASES else ()
        ),
    )


@pytest.mark.parametrize("operation", ("load", "store"))
def test_float_atomic_memory_opengl_publishes_word_storage(tmp_path, operation):
    report = _report(tmp_path, _source(operation, "member"), "opengl")
    assert report["summary"]["failedCount"] == 0
    generated = (tmp_path / report["artifacts"][0]["path"]).read_text()
    assert "crossgl_words_Counter" in generated
    assert "uintBitsToFloat(atomicOr(" in generated
    assert "crossgl_pending_float_atomic" not in generated


def _resource_request(root, target, operation):
    source, descriptor, package = _package(
        root,
        target,
        "float",
        (1, 1, 1),
        source=_resource_source(operation, "float").replace(
            "const device float* values", "device float* values"
        ),
        software_subgroups=False,
    )
    updates = [1.25, -2.5, 3.75, -4.0]
    initial = [-5.5, 6.25, -7.75, 8.5]
    guard = 42.25

    def typed(values, aggregate=False):
        return {
            "dtype": "float32",
            "shape": [len(values), 1] if aggregate else [len(values)],
            "values": values,
        }

    inputs = {
        "values": typed(updates),
        "results": typed([guard, *initial, guard, guard], True),
        "alternate": typed([guard, 20.0, 21.0, 22.0, 23.0, guard, guard], True),
        "observed": typed([guard] * 6),
    }
    outputs = {
        "values": inputs["values"],
        "alternate": inputs["alternate"],
        "results": typed(
            [
                guard,
                *[a + b if operation == "add" else b for a, b in zip(initial, updates)],
                guard,
                guard,
            ],
            True,
        ),
        "observed": typed([*initial, guard, guard]),
    }
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    (root / "workload.json").write_text(
        json.dumps({"operation": operation, "case": "resource-arithmetic"}),
        encoding="utf-8",
    )
    return source, request, expected


@pytest.mark.parametrize("operation", ("add", "exchange"))
@pytest.mark.parametrize("target", ("metal", "directx"))
def test_float_atomic_memory_resource_arithmetic_compiles(tmp_path, operation, target):
    _, request, _ = _resource_request(tmp_path, target, operation)
    if shutil.which("xcrun" if target == "metal" else "dxc"):
        _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
        assert module.stat().st_size


@pytest.mark.parametrize("operation", ("add", "exchange"))
def test_float_atomic_memory_resource_arithmetic_executes(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            f"set {REQUIRE_ENV}=1 for native floating resource atomic execution"
        )
    assert sys.platform in {"darwin", "win32"}
    target = "metal" if sys.platform == "darwin" else "directx"
    source, request, expected = _resource_request(tmp_path, target, operation)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="resource_atomic",
    )


def test_float_atomic_memory_requires_native_metal_and_directx_evidence():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_float_atomic_memory.py" in step
        assert "test_mlx_float_atomic_memory.py" in step
        assert "--timeout-seconds 1200" in step
        assert "if:" not in step and "continue-on-error" not in workflow
    opengl = ci_coverage.workflow_step_section(
        workflow, "Validate indexed OpenGL gather and resource aggregates"
    )
    assert "test_float_atomic_memory.py" in opengl
    assert REQUIRE_ENV not in opengl

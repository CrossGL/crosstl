"""Atomic loads retain storage identity, values and address evaluation."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_backend.test_metal.test_codegen import convert
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_ATOMIC_LOAD_RUNTIME"
CASES = (
    "scalar",
    "member",
    "pointer",
    "helper",
    "conditional",
    "branch",
    "loop",
    "aggregate",
    "workgroup",
)
GUARD = 123456789


def _source(kind, case):
    storage = f"atomic_{kind}" if case in {"scalar", "pointer"} else "Counter"
    member = "" if case in {"scalar", "pointer"} else ".value"
    declarations = (
        f"struct Counter {{ {kind} before; atomic<{kind}> value; {kind} after; }};"
    )
    expression = (
        f"atomic_load_explicit(&values[position++]{member}, memory_order_relaxed)"
    )
    setup = ""
    statement = f"results[tid] = {expression};"
    if case == "pointer":
        setup = f"device atomic<{kind}>* source = values + position++;"
        statement = "results[tid] = atomic_load_explicit(source, memory_order_relaxed);"
    elif case == "helper":
        declarations += f"""{kind} load(device Counter* source, uint index) {{
            return metal::atomic_load_explicit(&source[index].value, metal::memory_order_relaxed);
        }}"""
        statement = "results[tid] = load(values, position++);"
    elif case == "conditional":
        statement = f"results[tid] = tid == 1u ? {kind}(17) : {expression};"
    elif case == "branch":
        statement = f"if (tid != 1u) {{ {statement} }}"
    elif case == "loop":
        statement = f"for (uint iteration = 0u; iteration < 2u; ++iteration) {{ {statement} --position; }}"
    elif case == "aggregate":
        declarations += f"""struct View {{ device Counter* source; }};
        {kind} load(const thread View& view, uint index) {{
            return atomic_load_explicit(&view.source[index].value, memory_order_relaxed);
        }}"""
        setup = "View view{values}; view.source += 2; position = tid;"
        statement = "results[tid] = load(view, position++);"
    elif case == "workgroup":
        setup = f"""threadgroup atomic_{kind} counter[1];
        if (tid == 0u) {{ atomic_store_explicit(&counter[0], {kind}(0), memory_order_relaxed); }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        atomic_fetch_add_explicit(&counter[0], {kind}(1), memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);"""
        statement = (
            "results[tid] = atomic_load_explicit(&counter[0], memory_order_relaxed);"
        )
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void load_values(device {storage}* values [[buffer(0)]],
                        device {kind}* results [[buffer(1)]],
                        device uint* counts [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {{
    uint position = tid + 2u;
    {setup}
    {statement}
    counts[tid] = position;
}}
"""


def _request(root, target, kind, case):
    width = 32 if case == "workgroup" else 1
    count = 32 if case == "workgroup" else 3
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
        [-7, 2147483647, -2147483648] if kind == "int" else [0, 2147483648, 4294967295]
    )
    stride = 1 if case in {"scalar", "pointer"} else 3
    initial = [GUARD] * (7 * stride)
    for index, value in enumerate(values):
        initial[(index + 2) * stride + (stride != 1)] = value
    result = [GUARD] * (count + 3)
    counts = [GUARD] * (count + 3)
    for index in range(count):
        if case == "workgroup":
            result[index] = count
        elif case != "branch" or index != 1:
            result[index] = (
                17 if case == "conditional" and index == 1 else values[index]
            )
        active = not (case in {"conditional", "branch"} and index == 1)
        counts[index] = (
            index
            + (0 if case == "aggregate" else 2)
            + int(active and case not in {"loop", "workgroup"})
        )
    dtype = f"{kind}32"
    input_values = {
        "dtype": dtype,
        "shape": [7] if stride == 1 else [7, stride],
        "values": initial,
    }
    inputs = {
        "values": input_values,
        "results": {
            "dtype": dtype,
            "shape": [count + 3],
            "values": [GUARD] * (count + 3),
        },
        "counts": {
            "dtype": "uint32",
            "shape": [count + 3],
            "values": [GUARD] * (count + 3),
        },
    }
    expected = _bound_values(
        descriptor,
        {
            "values": input_values,
            "results": {"dtype": dtype, "shape": [count + 3], "values": result},
            "counts": {"dtype": "uint32", "shape": [count + 3], "values": counts},
        },
    )
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
def test_atomic_load_translates(tmp_path, target, kind, case):
    _, request, _ = _request(tmp_path, target, kind, case)
    generated = request.artifact_path.read_text()
    assert {
        "metal": "atomic_load_explicit",
        "directx": "InterlockedOr",
        "opengl": "atomicOr",
    }[target] in generated
    assert "atomicLoad(" not in generated
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if shutil.which(tool):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_atomic_load_executes_natively(tmp_path, kind, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required atomic-load execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, kind, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="load_values",
    )


def _report(root, source, target):
    (root / "load.metal").write_text(source, encoding="utf-8")
    return translate_project(
        ProjectConfig(
            root=root,
            targets=(target,),
            include_patterns=("load.metal",),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "order",
    (
        "memory_order_acquire",
        "memory_order_seq_cst",
        "other::memory_order_relaxed",
        "order",
    ),
)
def test_atomic_load_order_is_not_silently_weakened(tmp_path, target, order):
    report = _report(
        tmp_path,
        _source("int", "member").replace("memory_order_relaxed", order),
        target,
    )
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metal-atomic-load-unsupported"
        for item in report["diagnostics"]
    )
    assert all(
        not (tmp_path / artifact["path"]).exists() for artifact in report["artifacts"]
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_atomic_load_shadowed_order_is_not_a_builtin(tmp_path, target):
    source = _source("int", "member").replace(
        "uint position = tid + 2u;",
        "const memory_order memory_order_relaxed = metal::memory_order_relaxed;\nuint position = tid + 2u;",
    )
    report = _report(tmp_path, source, target)
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metal-atomic-load-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_unsupported_atomic_load_type_does_not_publish_a_noop(tmp_path, target):
    report = _report(tmp_path, _source("ulong", "member"), target)
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.metal-atomic-load-unsupported"
        and item["location"]["line"] > 1
        for item in report["diagnostics"]
    )
    assert all(
        not (tmp_path / artifact["path"]).exists() for artifact in report["artifacts"]
    )


def test_custom_atomic_load_name_is_not_rewritten():
    source = """namespace custom { int atomic_load_explicit(int value, int order) { return value + order; } }
    int exercise() { return custom::atomic_load_explicit(3, 4); }"""
    crossgl = convert(source)
    assert "atomicLoad(" not in crossgl
    assert "atomic_load_explicit" in crossgl


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_atomic_load_private_storage_is_rejected(tmp_path, target):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void load_value(device int* results [[buffer(0)]]) {
    atomic_int counter;
    results[0] = atomic_load_explicit(&counter, memory_order_relaxed);
}"""
    report = _report(tmp_path, source, target)
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.metal-atomic-load-unsupported"
        for item in report["diagnostics"]
    )
    assert all(
        not (tmp_path / artifact["path"]).exists() for artifact in report["artifacts"]
    )


@pytest.mark.parametrize(
    "arguments", ("&values[0].value", "&values[0].value, memory_order_relaxed, 0")
)
def test_atomic_load_rejects_unknown_argument_contract(tmp_path, arguments):
    source = _source("int", "member").replace(
        "&values[position++].value, memory_order_relaxed", arguments
    )
    report = _report(tmp_path, source, "crossgl")
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.metal-atomic-load-unsupported"
        for item in report["diagnostics"]
    )


def test_atomic_load_transport_collision_is_not_silently_rebound(tmp_path):
    source = _source("int", "member").replace(
        "kernel void", "int atomicLoad(int value) { return value + 1; }\nkernel void"
    )
    report = _report(tmp_path, source, "crossgl")
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.metal-atomic-load-unsupported"
        and "conflicts" in item["message"]
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_atomic_load_readonly_storage_contract(tmp_path, target):
    source = _source("int", "member").replace(
        "device Counter* values", "const device Counter* values"
    )
    report = _report(tmp_path, source, target)
    if target == "metal":
        assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
        generated = (tmp_path / "out/metal/load.metal").read_text()
        assert "reinterpret_cast<const device atomic_int*>" in generated
        if shutil.which("xcrun"):
            _, module = _compile(generated, target, tmp_path)
            assert module.stat().st_size
    else:
        assert report["summary"]["failedCount"] == 1
        assert all(
            not (tmp_path / artifact["path"]).exists()
            for artifact in report["artifacts"]
        )


def test_atomic_loads_are_required_on_each_native_target():
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
        assert "test_atomic_load_runtime.py" in step
        assert "test_mlx_atomic_load_runtime.py" in step
        assert "--timeout-seconds 1200" in step
        assert "if:" not in step and "continue-on-error" not in workflow

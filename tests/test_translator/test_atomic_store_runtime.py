"""Atomic stores retain storage identity, ordering and argument evaluation."""

import os
import shutil
import sys

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

REQUIRE_ENV = "CROSTL_REQUIRE_ATOMIC_STORE_RUNTIME"
CASES = (
    "scalar",
    "member",
    "pointer",
    "helper",
    "returned",
    "conditional",
    "branch",
    "workgroup",
)


def _source(kind, case):
    field = f"atomic<{kind}> value;"
    storage = "Counter"
    member = ".value"
    helper = ""
    if case in {"scalar", "pointer"}:
        storage, member = f"atomic_{kind}", ""
    elif case == "nested":
        field = f"struct Payload {{ atomic<{kind}> value; }};\nstruct Counter {{ {kind} before; Payload payload; {kind} after; }};"
        member = ".payload.value"
    elif case == "array":
        field = f"atomic<{kind}> value[2];"
        member = ".value[1]"
    if case != "nested":
        field = f"struct Counter {{ {kind} before; {field} {kind} after; }};"
    call = f"atomic_store_explicit(&results[position++]{member}, values[value_index++], memory_order_relaxed);"
    if case == "pointer":
        call = f"device atomic<{kind}>* destination = results + position++;\natomic_store_explicit(destination, values[value_index++], memory_order_relaxed);"
    elif case == "conditional":
        helper = f"""void store(device Counter* destination, uint index, {kind} value) {{
            return (index & 1u) != 0u
                ? atomic_store_explicit(&destination[index].value, value, memory_order_relaxed)
                : atomic_store_explicit(&destination[index].value, value + {kind}(1), memory_order_relaxed);
        }}"""
        call = "store(results, position++, values[value_index++]);"
    elif case in {"helper", "returned"}:
        returned = "return " if case == "returned" else ""
        helper = f"""void store(device Counter* destination, uint index, {kind} value) {{
            {returned}metal::atomic_store_explicit(&destination[index].value, value, metal::memory_order_relaxed);
        }}"""
        call = "store(results, position++, values[value_index++]);"
    elif case == "branch":
        call = f"if (tid != 1u) {{ {call} }}"
    elif case == "workgroup":
        call = f"""threadgroup atomic_{kind} temporary[1];
        atomic_store_explicit(&temporary[0], values[value_index++], memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        results[position++].value = atomic_fetch_add_explicit(&temporary[0], {kind}(0), memory_order_relaxed);"""
        # The result is ordinary storage; only the shared temporary is atomic.
        field = f"struct Counter {{ {kind} before; {kind} value; {kind} after; }};"
    return f"""#include <metal_stdlib>
using namespace metal;
{field}
{helper}
kernel void store_values(const device {kind}* values [[buffer(0)]],
                         device {storage}* results [[buffer(1)]],
                         device uint* counts [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {{
    uint position = tid + 2;
    uint value_index = tid;
    {call}
    counts[tid * 2] = position;
    counts[tid * 2 + 1] = value_index;
}}
"""


def _request(root, target, kind, case):
    source, descriptor, package = _package(
        root,
        target,
        kind,
        (1, 1, 1),
        source=_source(kind, case),
        software_subgroups=False,
    )
    values = (
        [-7, 2147483647, -2147483648] if kind == "int" else [0, 2147483648, 4294967295]
    )
    width = 1 if case in {"scalar", "pointer"} else 3
    member = 0 if width == 1 else 1
    initial = [123456789] * (7 * width)
    result = list(initial)
    counts = []
    for index, value in enumerate(values):
        active = case != "branch" or index != 1
        if case == "conditional" and index % 2 == 0:
            value += 1
            if kind == "uint":
                value %= 2**32
        if active:
            result[(index + 2) * width + member] = value
        counts.extend([index + 2 + int(active), index + int(active)])
    shape = [7] if width == 1 else [7, width]
    dtype = f"{kind}32"
    inputs = {
        "values": {"dtype": dtype, "shape": [3], "values": values},
        "results": {"dtype": dtype, "shape": shape, "values": initial},
        "counts": {"dtype": "uint32", "shape": [6], "values": [999] * 6},
    }
    outputs = _bound_values(
        descriptor,
        {
            "results": {"dtype": dtype, "shape": shape, "values": result},
            "counts": {"dtype": "uint32", "shape": [6], "values": counts},
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_atomic_store_translates(tmp_path, target, kind, case):
    _, request, _ = _request(tmp_path, target, kind, case)
    generated = request.artifact_path.read_text()
    intrinsic = {
        "metal": "atomic_store_explicit",
        "directx": "InterlockedExchange",
        "opengl": "atomicExchange",
    }[target]
    assert intrinsic in generated
    assert "atomicStore(" not in generated
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if shutil.which(tool):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_atomic_store_executes_natively(tmp_path, kind, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required atomic-store execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, kind, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="store_values",
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "order",
    (
        "memory_order_release",
        "memory_order_seq_cst",
        "other::memory_order_relaxed",
        "order",
    ),
)
def test_atomic_store_order_is_not_silently_weakened(tmp_path, target, order):
    (tmp_path / "store.metal").write_text(
        _source("int", "member").replace("memory_order_relaxed", order),
        encoding="utf-8",
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=("store.metal",),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metal-atomic-store-unsupported"
        for item in report["diagnostics"]
    )
    for artifact in report["artifacts"]:
        assert not (tmp_path / artifact["path"]).exists()


def test_custom_atomic_store_name_is_not_rewritten():
    source = """namespace custom { void atomic_store_explicit(thread int& value, int other, int order) { value = other + order; } }
    void exercise() { int value = 0; custom::atomic_store_explicit(value, 3, 4); }"""
    crossgl = convert(source)
    assert "atomicStore(" not in crossgl
    assert "atomic_store_explicit" in crossgl


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_local_memory_order_name_is_not_treated_as_a_builtin(tmp_path, target):
    source = _source("int", "member").replace(
        "uint position = tid + 2;",
        "const memory_order memory_order_relaxed = metal::memory_order_relaxed;\nuint position = tid + 2;",
    )
    (tmp_path / "store.metal").write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=("store.metal",),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metal-atomic-store-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_unsupported_atomic_store_type_does_not_publish_a_noop(tmp_path, target):
    (tmp_path / "store.metal").write_text(_source("ulong", "member"), encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=("store.metal",),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    assert any("atomicStore" in item["message"] for item in report["diagnostics"])
    for artifact in report["artifacts"]:
        assert not (tmp_path / artifact["path"]).exists()


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", ("nested", "array"))
def test_atomic_store_nested_storage_compiles(tmp_path, target, kind, case):
    _, _, package = _package(
        tmp_path,
        target,
        kind,
        (1, 1, 1),
        source=_source(kind, case),
        software_subgroups=False,
    )
    suffix = {"metal": "*.metal", "directx": "*.hlsl", "opengl": "*.glsl"}[target]
    (artifact,) = tuple((package / "artifacts").rglob(suffix))
    generated = artifact.read_text()
    assert "atomicStore(" not in generated
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if shutil.which(tool):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size

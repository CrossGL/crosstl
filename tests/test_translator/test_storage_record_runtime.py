"""Mixed storage records through packaging and native target execution."""

import copy
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    NativeLoaderDispatchError,
    RuntimeAllocationView,
    RuntimeValue,
    build_native_loader_dispatch_request,
    pack_storage_records,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_STORAGE_RECORD_RUNTIME"
CASES = {
    "leading-padding": [("int", "rank"), ("long", "offset")],
    "trailing-padding": [("long", "offset"), ("int", "rank")],
    "mixed-bits": [
        ("float", "weight"),
        ("long", "offset"),
        ("uint", "tag"),
        ("ulong", "count"),
    ],
    "reordered-bits": [
        ("ulong", "count"),
        ("long", "offset"),
        ("uint", "tag"),
        ("float", "weight"),
    ],
    "matrix-parameters": [
        (t, n)
        for t, names in [
            ("int", ("M", "N", "K", "lda", "ldb", "ldd", "tiles_n", "tiles_m")),
            ("long", ("batch_stride_a", "batch_stride_b", "batch_stride_d")),
            ("int", ("swizzle_log", "gemm_k_iterations_aligned", "batch_ndim")),
        ]
        for n in names
    ],
}


def _source(case):
    fields = CASES[case]
    body = " ".join(f"{dtype} {name};" for dtype, name in fields)
    assignments = "\n".join(
        f"results[j].{name} = values[i].{name};" for _, name in fields
    )
    changed = "batch_stride_a" if case == "matrix-parameters" else "offset"
    return f"""#include <metal_stdlib>
using namespace metal;
struct Metadata {{ {body} }};
kernel void record_storage(constant Metadata* values [[buffer(0)]],
                           device Metadata* results [[buffer(1)]],
                           constant long* deltas [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {{
    uint i = tid & 3u;
    uint j = i + 1u;
    long delta = deltas[0];
    {assignments}
    results[j].{changed} = values[i].{changed} + delta;
}}
"""


def _request(root, target, case, offset=False, *, fault=None):
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    layouts = {
        b["scalarLayout"].get("memberName", b["name"]): b["scalarLayout"]
        for b in descriptor["bindings"]
    }
    layout = layouts["values"]
    assert layout["elementType"] == "record"
    assert layout["structMembers"] == layouts["results"]["structMembers"]
    words_per_record = layout["elementStrideBytes"] // 4
    fields = CASES[case]
    changed = "batch_stride_a" if case == "matrix-parameters" else "offset"
    records = []
    for i in range(4):
        record = {}
        for dtype, name in fields:
            record[name] = {
                "int": [-2147483648, -7, 0, 2147483647][i],
                "uint": [0, 2**32 - 1, 0x80000000, 0x5A1B2C3D][i],
                "long": [-(2**63), -4294967298, 4294967297, 2**63 - 4294967298][i],
                "ulong": [0, 2**64 - 1, 2**63, 4294967297][i],
                "float": [-0.0, 1.5, -2.5, 0.0][i],
            }[dtype]
        records.append(record)
    input_words = pack_storage_records(layout, records)
    expected_records = copy.deepcopy(records)
    for record in expected_records:
        record[changed] += 4294967297
    expected_words = pack_storage_records(layout, expected_records)
    # Physical float transport also exercises a quiet NaN payload without conversion.
    if any(name == "weight" for _, name in fields):
        weight = next(
            m["offsetBytes"] // 4
            for m in layout["structMembers"]
            if m["name"] == "weight"
        )
        input_words[3 * words_per_record + weight] = 0x7FC12345
        expected_words[3 * words_per_record + weight] = 0x7FC12345
    guard = [0x5A1B2C3D] * words_per_record
    expected = guard + expected_words + guard
    initial = guard + [0] * (4 * words_per_record) + guard

    def value(name, words, count):
        size = len(words) * 4
        byte_offset = layout["elementStrideBytes"] * 256 if offset else 0
        view = (
            RuntimeAllocationView(name, byte_offset, size, byte_offset + size + 256)
            if offset
            else None
        )
        return RuntimeValue(
            name=name,
            dtype="uint32",
            shape=(count, words_per_record),
            values=words,
            allocation=view,
        )

    inputs = {
        "values": value("values", input_words, 4),
        "results": value("results", initial, 6),
        "deltas": RuntimeValue(
            name="deltas", dtype="int64", shape=(1,), values=[4294967297]
        ),
    }
    outputs = {"results": value("results", expected, 6)}
    if fault is not None:
        old = inputs["values"]
        size = len(input_words) * 4
        view = {
            "misaligned": RuntimeAllocationView("values", 4, size, size + 256),
            "undersized": RuntimeAllocationView("values", 0, size - 4, size + 256),
        }[fault]
        inputs["values"] = replace(old, allocation=view)
    before = copy.deepcopy(input_words)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert input_words == before
    assert not request.execution_plan.diagnostics
    bound = next(
        r
        for r in request.execution_plan.resource_bindings
        if r.binding.access == "read"
    )
    assert bound.allocation.byte_length == len(input_words) * 4
    return request, _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [6, words_per_record],
                "values": expected,
            }
        },
    )


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("offset", (False, True))
def test_mixed_record_package_compiles(tmp_path, target, case, offset):
    request, _ = _request(tmp_path, target, case, offset)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("fault", ("misaligned", "undersized"))
def test_mixed_record_dispatch_rejects_invalid_views(tmp_path, target, fault):
    with pytest.raises(NativeLoaderDispatchError, match="execution-plan-invalid"):
        _request(tmp_path, target, "leading-padding", fault=fault)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("offset", (False, True))
def test_mixed_records_execute(tmp_path, case, offset):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required mixed-record execution")
    target = {"linux": "opengl", "darwin": "metal", "win32": "directx"}[sys.platform]
    request, expected = _request(tmp_path, target, case, offset)
    options = (
        {
            "original_source": _source(case),
            "original_entry": "record_storage",
            "metal_compile_flags": ("-Wall", "-Wextra", "-Werror"),
        }
        if target == "metal"
        else {}
    )
    _execute(request, expected, tmp_path, **options)


def test_mixed_records_are_required_in_native_ci():
    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    step = workflow.split("- name: Validate collective helper arguments", 1)[1].split(
        "- name:", 1
    )[0]
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_storage_record_runtime.py" in step

"""Aligned byte records copy backing words without changing their representation."""

import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_WORKGROUP_RECORD_TRANSFERS"
CASES = [
    (kind, mode)
    for kind in ("uint", "int", "float")
    for mode in ("direct", "offset", "helper", "forward", "small", "loop", "alias")
]
GUARD = 0x5A1B2C3D
WORDS = [
    0,
    0x3F800000,
    0x80000000,
    0x3F012345,
    0x7F7FFFFF,
    0xBF000000,
    0x7FC00123,
    0x00000001,
]


def _source(kind, mode):
    count = 1 if mode == "small" else 8 if mode == "loop" else 4
    record_words = min(count, 4)
    alignment = 1 if mode == "small" else 4
    destination = "destination" if mode in {"helper", "forward"} else "&tile[2]"
    source = "source" if mode in {"helper", "forward"} else "&inputs[base]"
    transfer = (
        f"*((threadgroup Holder::Bytes*)({destination})) = "
        f"*((const device Holder::Bytes*)({source}));"
    )
    helper = ""
    if mode in {"helper", "forward"}:
        helper = (
            f"void transfer(threadgroup {kind}* destination, "
            f"const device {kind}* source) {{ {transfer} }}"
        )
        transfer = "transfer(&tile[2], &inputs[base]);"
        if mode == "forward":
            helper += (
                f"\nvoid forward(threadgroup {kind}* destination, "
                f"const device {kind}* source) {{ transfer(destination + 1, source + 1); }}"
            )
            transfer = "forward(&tile[1], &inputs[base - 1]);"
    if mode == "offset":
        transfer = transfer.replace("&inputs[base]", "inputs + base")
    if mode == "loop":
        transfer = transfer.replace("tile[2]", "tile[2 + j]").replace(
            "inputs[base]", "inputs[base + j]"
        )
        transfer = f"for (uint j = 0u; j < 8u; j += 4u) {{ {transfer} }}"
    if mode == "alias":
        transfer = (
            f"threadgroup {kind}* destination = tile; destination += 2; "
            + transfer.replace("&tile[2]", "destination")
        )
    return f"""#include <metal_stdlib>
using namespace metal;
struct Holder {{ struct alignas({alignment}) Bytes {{ uchar data[{record_words * 4}]; }}; }};
{helper}
kernel void record_transfer(const device {kind}* inputs [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {{
    threadgroup {kind} tile[{count + 4}];
    uint base = 4u + {count}u * tid;
    for (uint i = 0u; i < {count + 4}u; ++i) {{ tile[i] = as_type<{kind}>(0x5A1B2C3Du); }}
    {transfer}
    for (uint i = 0u; i < {count + 4}u; ++i) {{
        results[4u + {count + 4}u * tid + i] = as_type<uint>(tile[i]);
    }}
}}
"""


def _request(
    root,
    target,
    case,
    *,
    source=None,
    workgroup_access_assertions=(),
    index_range_assertions=(),
):
    kind, mode = case
    count = 1 if mode == "small" else 8 if mode == "loop" else 4
    words = [WORDS[i % len(WORDS)] for i in range(4 + count * 5)]
    if kind == "float":
        words = [0x00800123 if word == 0x7FC00123 else word for word in words]
    fmt = {"uint": "I", "int": "i", "float": "f"}[kind]
    values = [struct.unpack("<" + fmt, struct.pack("<I", word))[0] for word in words]
    expected_words = [GUARD] * 4
    for tid in range(5):
        expected_words.extend(
            [GUARD] * 2 + words[4 + count * tid : 4 + count * (tid + 1)] + [GUARD] * 2
        )
    expected_words.extend([GUARD] * 4)
    source, descriptor, package = _package(
        root,
        target,
        kind,
        (1, 1, 1),
        source=source or _source(*case),
        software_subgroups=False,
        workgroup_access_assertions=workgroup_access_assertions,
        index_range_assertions=index_range_assertions,
    )
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {
                "dtype": {"uint": "uint32", "int": "int32", "float": "float32"}[kind],
                "shape": [len(values)],
                "values": values,
            },
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": (
                    [GUARD] * 4 + [0xDEADBEEF] * (len(expected_words) - 8) + [GUARD] * 4
                ),
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": expected_words,
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [5, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_record_transfer_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_record_transfer_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required record-transfer execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="record_transfer",
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


def test_record_transfer_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_workgroup_record_transfers.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


@pytest.mark.parametrize("mode", ["direct", "helper", "forward", "loop", "alias"])
def test_record_transfer_checks_entire_destination_span(tmp_path, mode):
    source = _source("uint", mode)
    size = 12 if mode == "loop" else 8
    source = source.replace(f"tile[{size}]", f"tile[{size - 3}]")
    (tmp_path / "transfer.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("transfer.metal",),
            targets=("directx",),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    diagnostic = next(
        item for item in report["diagnostics"] if item["severity"] == "error"
    )
    assert (
        diagnostic["code"] == "project.translate.directx-workgroup-pointer-unsupported"
    )
    assert "extent" in diagnostic["message"]


@pytest.mark.parametrize(
    "old,new,reason",
    [
        (
            "uchar data[16]",
            "uint data[4]",
            "unsupported-storage-workgroup-aggregate-layout",
        ),
        (
            "alignas(4)",
            "alignas(8)",
            "unsupported-storage-workgroup-aggregate-alignment",
        ),
        (
            "uchar data[16]",
            "uchar data[15]",
            "unsupported-storage-workgroup-aggregate-layout",
        ),
        (
            "const device uint* inputs",
            "const device int* inputs",
            "storage-workgroup-aggregate-backing-mismatch",
        ),
        (
            "&inputs[base]",
            "&inputs[base++]",
            "storage-workgroup-aggregate-offset-unprovable",
        ),
        (
            "threadgroup Holder::Bytes*",
            "const threadgroup Holder::Bytes*",
            "storage-workgroup-aggregate-destination-read-only",
        ),
    ],
)
def test_record_transfer_rejects_unproven_layouts_and_access(
    tmp_path, old, new, reason
):
    (tmp_path / "transfer.metal").write_text(
        _source("uint", "direct").replace(old, new)
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("transfer.metal",),
            targets=("directx",),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    diagnostic = next(
        item for item in report["diagnostics"] if item["severity"] == "error"
    )
    assert diagnostic["code"] == "project.translate.pointer-reinterpret-unsupported"
    assert diagnostic["details"]["pointerReinterpretation"]["reason"] == reason


@pytest.mark.parametrize(
    "mode", ["update", "shadow", "dynamic-bound", "reference-call"]
)
def test_record_transfer_does_not_assume_modified_loop_indices(tmp_path, mode):
    source = _source("uint", "loop")
    start = source.index("*((threadgroup Holder::Bytes*)")
    end = source.index(";", start) + 1
    transfer = source[start:end]
    if mode == "shadow":
        source = (
            source[:start] + "{ uint j = inputs[0]; " + transfer + " }" + source[end:]
        )
    elif mode == "dynamic-bound":
        source = source.replace("j < 8u", "j < inputs[0]")
    elif mode == "reference-call":
        source = source[:start] + "advance(j); " + transfer + source[end:]
        source = source.replace(
            "kernel void",
            "void advance(thread uint& value) { value += 8u; }\nkernel void",
        )
    else:
        source = source[:start] + "j += 4u; " + transfer + source[end:]
    (tmp_path / "transfer.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("transfer.metal",),
            targets=("directx",),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.directx-workgroup-pointer-unsupported"
        for item in report["diagnostics"]
    )

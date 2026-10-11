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
    for mode in (
        "direct",
        "offset",
        "helper",
        "forward",
        "small",
        "loop",
        "alias",
        "wide-direct",
        "wide-forward",
    )
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
    wide = mode.startswith("wide-")
    if wide:
        mode = mode[len("wide-") :]
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
    setup = ""
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
        setup = f"threadgroup {kind}* destination = tile; destination += 2;"
        transfer = transfer.replace("&tile[2]", "destination")
    text = f"""#include <metal_stdlib>
using namespace metal;
struct Holder {{ struct alignas({alignment}) Bytes {{ uchar data[{record_words * 4}]; }}; }};
{helper}
kernel void record_transfer(const device {kind}* inputs [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {{
    threadgroup {kind} tile[{count + 4}];
    {setup}
    if (tid < 5u) {{
        uint base = 4u + {count}u * tid;
        for (uint i = 0u; i < {count + 4}u; ++i) {{ tile[i] = as_type<{kind}>(0x5A1B2C3Du); }}
        {transfer}
        for (uint i = 0u; i < {count + 4}u; ++i) {{
            results[4u + {count + 4}u * tid + i] = as_type<uint>(tile[i]);
        }}
    }}
}}
"""
    if wide:
        text = text.replace(
            f"uint base = 4u + {count}u * tid;",
            f"ulong base = 4ul + {count}ul * ulong(tid);",
        )
    return text


def _source_span_report(root, expression, words=2, *, guard=None, helper=False):
    transfer = "*((threadgroup Bytes*)tile) = *((const device Bytes*)source);"
    declaration = ""
    if helper:
        declaration = (
            "void copy_words(threadgroup uint* tile, const device uint* source) { "
            + transfer
            + " }"
        )
        transfer = f"copy_words(tile, inputs + ({expression}));"
    else:
        transfer = transfer.replace("*)source)", f"*)(inputs + ({expression})))")
    transfer += " results[0] = tile[0];"
    if guard:
        transfer = f"if ({guard}) {{ {transfer} }}"
    source = f"""#include <metal_stdlib>
using namespace metal;
struct alignas(4) Bytes {{ uchar bytes[{words * 4}]; }};
{declaration}
kernel void copy_span(const device uint* inputs [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      constant ulong& offset [[buffer(2)]]) {{
    threadgroup uint tile[{words}];
    {transfer}
}}
"""
    (root / "copy.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("copy.metal",),
            targets=("opengl",),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    return report.to_json()


@pytest.mark.parametrize("words", [1, 2, 4, 8])
@pytest.mark.parametrize("suffix", ["", "u", "l", "ul"])
@pytest.mark.parametrize("overflow", [False, True])
def test_record_transfer_checks_complete_source_span(tmp_path, words, suffix, overflow):
    start = (1 << 31) - words + int(overflow)
    report = _source_span_report(tmp_path, f"{start}{suffix}", words)
    if overflow:
        assert report["summary"]["translatedCount"] == 0
        diagnostic = next(
            item for item in report["diagnostics"] if item["severity"] == "error"
        )
        assert diagnostic["code"] == "project.translate.opengl-index-type-unsupported"
        details = diagnostic["details"]["indexConversion"]
        assert details["reason"] == "constant-index-out-of-range"
        assert details["sourceRange"]["maximum"] == 1 << 31
        assert details["acceptedRange"]["maximum"] == (1 << 31) - 1
        assert diagnostic["location"]["file"] == "copy.metal"
    else:
        assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
        artifact = next(
            item for item in report["artifacts"] if item["status"] == "translated"
        )
        _compile((tmp_path / artifact["path"]).read_text(), "opengl", tmp_path)


@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize(
    "expression,guard,accepted",
    [
        ("offset & 15ul", None, True),
        ("offset % 16ul", None, True),
        (
            "(offset + 18446744073709551615ul) / 18446744073709551615ul",
            "offset <= 3ul",
            False,
        ),
        ("offset", "offset <= 2147483646ul", True),
        ("offset", "offset <= 2147483647ul", False),
        ("offset", None, False),
        ("uint(offset)", None, False),
        ("4294967296ul", None, False),
        ("18446744073709551615ul", None, False),
        ("-1l", None, False),
    ],
)
def test_record_transfer_proves_dynamic_source_spans(
    tmp_path, expression, guard, accepted, helper
):
    report = _source_span_report(tmp_path, expression, guard=guard, helper=helper)
    if accepted:
        assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
        artifact = next(
            item for item in report["artifacts"] if item["status"] == "translated"
        )
        _compile((tmp_path / artifact["path"]).read_text(), "opengl", tmp_path)
    else:
        assert report["summary"]["translatedCount"] == 0
        assert any(
            item["code"] == "project.translate.opengl-index-type-unsupported"
            for item in report["diagnostics"]
        )


@pytest.mark.parametrize("mode", ["mixed-call", "update", "unknown-update", "shadow"])
def test_record_transfer_invalidates_source_offset_proofs(tmp_path, mode):
    source = _source("uint", "helper")
    if mode == "mixed-call":
        source = source.replace(
            "transfer(&tile[2], &inputs[base]);",
            "transfer(&tile[2], &inputs[base]); transfer(&tile[2], &inputs[inputs[0]]);",
        )
    else:
        prefix = {
            "update": "source += 2147483647; ",
            "unknown-update": "source += source[0]; ",
            "shadow": (
                "const device uint* saved = source; { const device uint* source = saved + 2147483647; "
            ),
        }[mode]
        start = source.index("*((threadgroup Holder::Bytes*)")
        end = source.index(";", start) + 1
        source = (
            source[:start]
            + prefix
            + source[start:end]
            + (" }" if mode == "shadow" else "")
            + source[end:]
        )
    (tmp_path / "copy.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("copy.metal",),
            targets=("opengl",),
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


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
        {"workgroupCount": [6, 1, 1], "workgroupSize": [1, 1, 1]},
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

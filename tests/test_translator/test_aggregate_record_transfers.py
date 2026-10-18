"""Typed record copies retain storage identity through private pointer fields."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig, translate_project
from tests.test_translator import test_workgroup_record_transfers as records
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_AGGREGATE_RECORD_TRANSFERS"
CASES = [
    (kind, mode)
    for kind in ("uint", "int", "float")
    for mode in ("direct", "helper", "nested", "rebase", "loop", "selection")
]


def _source(kind, mode):
    source = records._source(kind, "loop" if mode == "loop" else "direct")
    declarations = (
        f"struct Cursor {{ const device {kind}* src; threadgroup {kind}* dst; }};"
    )
    setup = "Cursor cursor{inputs + base, tile + 2};"
    source = source.replace("&tile[2 + j]", "&cursor.dst[j]").replace(
        "&inputs[base + j]", "&cursor.src[j]"
    )
    source = source.replace("&tile[2]", "cursor.dst").replace(
        "&inputs[base]", "cursor.src"
    )
    if mode == "nested":
        declarations += "struct Envelope { Cursor cursor; };"
        setup = "Envelope envelope{{inputs + base, tile + 2}};"
        source = source.replace("cursor.dst", "envelope.cursor.dst").replace(
            "cursor.src", "envelope.cursor.src"
        )
    elif mode == "rebase":
        setup = "Cursor cursor{inputs + base + 1, tile + 1}; cursor.src -= 1; cursor.dst += 1;"
    elif mode == "selection":
        declarations = f"struct Cursor {{ const device {kind}* src; threadgroup {kind}* dst; threadgroup {kind}* inspect; }};"
        setup = "Cursor cursor{inputs + base, tile + 2, tile}; if ((tid & 1u) != 0u) { cursor.dst = other + 2; cursor.inspect = other; }"
        source = source.replace(
            f"threadgroup {kind} tile[8];",
            f"threadgroup {kind} tile[8]; threadgroup {kind} other[8];",
        )
        source = source.replace(
            "(0x5A1B2C3Du); }",
            f"(0x5A1B2C3Du); other[i] = as_type<{kind}>(0x5A1B2C3Du); }}",
        )
        source = source.replace(
            "as_type<uint>(tile[i])", "as_type<uint>(cursor.inspect[i])"
        )
    elif mode == "helper":
        operation = "*((threadgroup Holder::Bytes*)(cursor.dst)) = *((const device Holder::Bytes*)(cursor.src));"
        assert operation in source
        source = source.replace(operation, "transfer(cursor);")
        declarations += f"void transfer(Cursor cursor) {{ {operation} }}"
    source = source.replace(
        "kernel void record_transfer", declarations + "\nkernel void record_transfer"
    )
    return (
        source.replace("    *((threadgroup", "    " + setup + "\n    *((threadgroup")
        if mode != "loop" and mode != "helper"
        else source.replace(
            "    for (uint j" if mode == "loop" else "    transfer(cursor);",
            "    "
            + setup
            + "\n"
            + ("    for (uint j" if mode == "loop" else "    transfer(cursor);"),
        )
    )


def _assertions(mode):
    return [
        {
            "entry_point": "record_transfer",
            "function": "transfer" if mode == "helper" else "record_transfer",
            "parameter": "envelope.cursor.dst" if mode == "nested" else "cursor.dst",
            "minimum": 2,
            "maximum": 9 if mode == "loop" else 5,
        }
    ]


def _request(root, target, case):
    kind, mode = case
    count = 8 if mode == "loop" else 4
    assertions = _assertions(mode) if target != "metal" else ()
    ranges = []
    if target == "opengl":
        ranges = [
            {
                "function": f"crosstl_workgroup_copy_{kind}",
                "expression": f"source.offset + {word}",
                "minimum": 4 + word,
                "maximum": 4 + count * 5 - min(count, 4) + word,
            }
            for word in range(4)
        ]
        ranges += [
            {
                "function": f"crosstl_{storage}_{operation}_{element}{suffix}",
                "expression": "reference.offset + index",
                "minimum": 0,
                "maximum": (
                    count + 3 if storage == "workgroup" else 4 + 5 * (count + 4) - 1
                ),
            }
            for storage in ("resource", "workgroup")
            for operation in ("load", "store")
            for element in dict.fromkeys((kind, "uint"))
            for suffix in ("", "_2")
        ]
    return records._request(
        root,
        target,
        (kind, "loop" if mode == "loop" else "direct"),
        source=_source(*case),
        workgroup_access_assertions=assertions,
        index_range_assertions=ranges,
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_aggregate_record_copy_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_aggregate_record_copy_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required aggregate record-copy execution")
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


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "record-copy-destination-range-unproven"),
        ("wrong-entry", "record-copy-destination-range-unproven"),
        ("wrong-function", "record-copy-destination-range-unproven"),
        ("wrong-member", "record-copy-destination-range-unproven"),
        ("past-end", "record-copy-destination-out-of-bounds"),
        ("negative", "record-copy-destination-out-of-bounds"),
        ("short", "record-copy-conflicting-access-assertions"),
        ("conflict", "record-copy-conflicting-access-assertions"),
    ],
)
def test_aggregate_record_copy_rejects_unproved_span(tmp_path, target, case, reason):
    (tmp_path / "transfer.metal").write_text(_source("uint", "direct"))
    assertions = _assertions("direct")
    if case == "missing":
        assertions = []
    elif case.startswith("wrong-"):
        assertions[0][
            {
                "wrong-entry": "entry_point",
                "wrong-function": "function",
                "wrong-member": "parameter",
            }[case]
        ] = "absent"
    elif case == "past-end":
        assertions[0]["maximum"] = 8
    elif case == "negative":
        assertions[0]["minimum"] = -1
    elif case == "short":
        assertions[0]["maximum"] = 4
    else:
        assertions.append({**assertions[0], "minimum": 6, "maximum": 7})
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("transfer.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
            workgroup_access_assertions=assertions,
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.resource-aggregate-unsupported"
        and reason in item["message"]
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "old,new,reason",
    [
        ("uchar data[16]", "uchar data[15]", "record-copy-layout"),
        ("alignas(4)", "alignas(8)", "record-copy-backing-layout"),
        ("uchar data[16]", "uint data[4]", "record-copy-layout"),
        ("uchar data[16]", "uchar data[16]; uint extra", "record-copy-layout"),
        ("(cursor.src)", "(cursor.src++)", "record-copy-address-side-effects"),
        ("(cursor.dst)", "(cursor.dst++)", "record-copy-address-side-effects"),
        (
            "threadgroup Holder::Bytes*",
            "const threadgroup Holder::Bytes*",
            "record-copy-access-contract",
        ),
    ],
)
def test_aggregate_record_copy_rejects_unsupported_layouts(
    tmp_path, target, old, new, reason
):
    source = _source("uint", "direct")
    assert old in source
    (tmp_path / "transfer.metal").write_text(source.replace(old, new))
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("transfer.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
            workgroup_access_assertions=_assertions("direct"),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.resource-aggregate-unsupported"
        and reason in item["message"]
        for item in report["diagnostics"]
    )


def test_aggregate_record_copy_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_aggregate_record_transfers.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step

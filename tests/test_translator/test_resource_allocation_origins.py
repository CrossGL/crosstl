"""Record-copy contracts use proven origins, not every same-element allocation."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig, translate_project
from tests.test_translator import test_workgroup_record_transfers as records
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_RESOURCE_ALLOCATION_ORIGINS"
CASES = [
    (kind, mode)
    for kind in ("uint", "int", "float")
    for mode in (
        "aggregate",
        "constructor",
        "returned",
        "forward",
        "pointer-return",
        "pointer-reference",
        "array",
        "shadow",
        "record-metadata",
    )
]


def _source(kind, mode):
    declarations = "struct Holder { struct alignas(4) Bytes { uchar data[16]; }; };"
    if mode == "record-metadata":
        declarations = declarations.replace(
            "uchar data[16];",
            "static constant constexpr uint size = 16u; uchar data[16];",
        )
    for name in ("LargeCursor", "SmallCursor"):
        constructor = (
            f"{name}(const device {kind}* s, threadgroup {kind}* d): src(s), dst(d) {{}}"
            if mode == "constructor"
            else ""
        )
        declarations += f"struct {name} {{ const device {kind}* src; threadgroup {kind}* dst; {constructor} }};"
    for size in ("large", "small"):
        declarations += f"""
void copy_{size}({size.capitalize()}Cursor cursor) {{
    *((threadgroup Holder::Bytes*)(cursor.dst)) = *((const device Holder::Bytes*)(cursor.src));
}}
"""
    setup = "LargeCursor large{inputs + base, tile + 2}; SmallCursor compact{inputs + base + 4, small};"
    operation = "copy_large(large); copy_small(compact);"
    if mode == "returned":
        declarations += f"LargeCursor make_large(const device {kind}* s, threadgroup {kind}* d) {{ LargeCursor cursor{{s, d}}; return cursor; }}"
        setup = "LargeCursor large = make_large(inputs + base, tile + 2); SmallCursor compact{inputs + base + 4, small};"
    elif mode == "forward":
        declarations += "void relay(LargeCursor first, SmallCursor second) { LargeCursor snapshot = first; copy_large(snapshot); copy_small(second); }"
        operation = "relay(large, compact);"
    elif mode == "pointer-return":
        declarations += f"threadgroup {kind}* forward_pointer(threadgroup {kind}* p) {{ return p + 1; }}"
        setup = setup.replace("tile + 2", "forward_pointer(tile + 1)")
    elif mode == "shadow":
        setup = (
            f"threadgroup {kind}* pointer = tile + 2; {{ threadgroup {kind}* pointer = small; pointer[0] = {kind}(0); }}"
            + setup.replace("tile + 2", "pointer")
        )
    elif mode == "pointer-reference":
        declarations += f"void select_pointer(threadgroup {kind}*& p, threadgroup {kind}* target) {{ p = target; }}"
        setup = setup.replace("tile + 2", "tile + 1")
        setup += "select_pointer(large.dst, tile + 2);"
    elif mode == "array":
        setup = (
            f"threadgroup {kind}* choices[2] = {{tile + 1, tile + 2}};"
            + setup.replace("tile + 2", "choices[tid & 1u] + (1u - (tid & 1u))")
        )
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void record_transfer(const device {kind}* inputs [[buffer(0)]], device uint* results [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{
    threadgroup {kind} tile[12];
    threadgroup {kind} small[4];
    uint base = 4u + 8u * tid;
    for (uint i = 0u; i < 12u; ++i) {{ tile[i] = as_type<{kind}>(0x5A1B2C3Du); }}
    {setup}
    {operation}
    for (uint i = 0u; i < 4u; ++i) {{ tile[6u + i] = small[i]; }}
    for (uint i = 0u; i < 12u; ++i) {{ results[4u + 12u * tid + i] = as_type<uint>(tile[i]); }}
}}
"""


def _assertions():
    return [
        {
            "entry_point": "record_transfer",
            "function": f"copy_{size}",
            "parameter": "cursor.dst",
            "minimum": start,
            "maximum": start + 3,
        }
        for size, start in (("large", 2), ("small", 0))
    ]


def _request(root, target, case):
    kind, mode = case
    ranges = []
    if target == "opengl":
        ranges = [
            {
                "function": f"crosstl_workgroup_copy_{kind}{suffix}",
                "expression": f"source.offset + {word}",
                "minimum": 4 + word,
                "maximum": 40 + word,
            }
            for suffix in ("", "_2")
            for word in range(4)
        ]
        ranges += [
            {
                "function": f"crosstl_{storage}_{operation}_{element}{suffix}",
                "expression": "reference.offset + index",
                "minimum": 0,
                "maximum": (
                    (
                        3
                        if (operation == "load" and suffix == "")
                        or (operation == "store" and suffix == "_2")
                        else 11
                    )
                    if storage == "workgroup"
                    else 63
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
        (kind, "loop"),
        source=_source(*case),
        workgroup_access_assertions=_assertions() if target != "metal" else (),
        index_range_assertions=ranges,
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_allocation_origins_compile(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_allocation_origins_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required allocation-origin execution")
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
    "mode",
    ["conditional", "reference", "pointer-reference", "array", "unknown", "same-type"],
)
def test_possible_small_allocation_is_not_excluded(tmp_path, target, mode):
    source = _source("uint", "aggregate")
    declaration = ""
    mutation = "if ((tid & 1u) != 0u) { large.dst = small; }"
    if mode == "reference":
        declaration = "void change(LargeCursor& cursor, threadgroup uint* target) { cursor.dst = target; }"
        mutation = "change(large, small);"
    elif mode == "pointer-reference":
        declaration = "void change(threadgroup uint*& slot, threadgroup uint* target) { slot = target; }"
        mutation = "change(large.dst, small);"
    elif mode == "array":
        mutation = "threadgroup uint* choices[2] = {tile + 2, small}; large.dst = choices[tid & 1u];"
    elif mode == "unknown":
        mutation = "threadgroup uint* unknown; large.dst = unknown;"
    elif mode == "same-type":
        mutation = "LargeCursor other{inputs + base, small}; copy_large(other);"
    source = source.replace(
        "kernel void record_transfer", declaration + "\nkernel void record_transfer"
    )
    source = source.replace(
        "    copy_large(large);", "    " + mutation + "\n    copy_large(large);"
    )
    (tmp_path / "transfer.metal").write_text(source)
    assertions = _assertions()
    assertions[0]["minimum"] = 0
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
        and "record-copy-destination-out-of-bounds" in item["message"]
        for item in report["diagnostics"]
    )


def test_allocation_origin_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_resource_allocation_origins.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step

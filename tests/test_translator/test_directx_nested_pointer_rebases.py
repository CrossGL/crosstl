"""Entry pointer rebases inside loops and branches keep one HLSL offset.

HLSL lowers a rebased Metal entry pointer to its buffer plus an ``int64_t``
logical offset. When the first rebase sits inside a loop, or inside a branch
whose pointer is used after the branch, the offset must be declared once at
entry: a declaration at the rebase would restart at zero on every iteration,
or go out of scope before the later access, and the access would silently
read the unrebased element.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.project.directx_toolchain import (
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
)

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void select_value(const device uint* input [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         device uint* output [[buffer(2)]]) {
    BODY
}
"""
DECLARATION = "int64_t input_offset = int64_t(0);"

HOISTED = {
    "loop": (
        "for (int i = 0; i < 3; ++i) { input += 2; } output[0] = input[1];",
        ["input_offset += int(2);"],
        ["output[0] = input.Load(uint((input_offset + 1)));"],
    ),
    "branch": (
        "if (count != 0) { input += 2; } else { input += 4; } output[0] = input[1];",
        ["input_offset += int(2);", "input_offset += int(4);"],
        ["output[0] = input.Load(uint((input_offset + 1)));"],
    ),
    "while": (
        "uint i = 0; while (i < count) { input += 1; i++; } output[0] = input[0];",
        ["input_offset += int(1);"],
        ["output[0] = input.Load(uint((input_offset + 0)));"],
    ),
    "branch-in-loop": (
        "for (uint i = 0; i < count; ++i) {"
        " if (i == 1) { input += 3; } output[i] = input[0]; }",
        ["input_offset += int(3);"],
        ["output[i] = input.Load(uint((input_offset + 0)));"],
    ),
    # The attention-vector mask shape: a guarded rebase before the loop and a
    # guarded advance inside it must update the same offset.
    "guarded-advance": (
        "if (count > 1) { input += count; }"
        " for (uint i = 0; i < 2; ++i) {"
        " output[i] = input[0]; if (count > 1) { input += 2; } }",
        ["input_offset += uint(select_value_count);", "input_offset += int(2);"],
        ["output[i] = input.Load(uint((input_offset + 0)));"],
    ),
}


def _entry(tmp_path, body):
    (tmp_path / "kernel.metal").write_text(SOURCE.replace("BODY", body))
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=("kernel.metal",),
        targets=("directx",),
        workgroup_size=(1, 1, 1),
        entry_points={"kernel.metal": ("select_value",)},
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert payload["diagnostics"] == []
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text("utf-8")
    entry = generated[generated.index("void CSMain") :]
    return generated, [line.strip() for line in entry.splitlines()]


def _compile(generated, tmp_path):
    compiler = shutil.which("dxc")
    if compiler is None:
        return
    source_path = tmp_path / "nested_rebase.hlsl"
    source_path.write_text(generated, encoding="utf-8")
    command = [compiler, "-T", dxc_profile_for_source("cs_6_6", generated)]
    command += [*dxc_compiler_arguments_for_source(generated), "-E", "CSMain"]
    command += [str(source_path), "-Fo", str(tmp_path / "nested_rebase.dxil")]
    compiled = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr


@pytest.mark.parametrize("case", HOISTED)
def test_nested_first_rebase_updates_one_entry_offset(tmp_path, case):
    body, updates, accesses = HOISTED[case]
    generated, lines = _entry(tmp_path, body)
    # One declaration, in the entry prologue, before any control statement.
    assert lines.count(DECLARATION) == 1
    assert lines[1] == DECLARATION
    assert not re.search(r"\binput_offset_\w*", generated)
    for statement in [*updates, *accesses]:
        assert statement in lines
    assert "input.Load(0)" not in generated
    assert "input.Load(1)" not in generated
    _compile(generated, tmp_path)


def test_branch_local_rebase_keeps_its_scoped_offset(tmp_path):
    # Every use of the rebased pointer is inside the branch that rebases it,
    # so the scoped declaration is exact and is left where it is.
    generated, lines = _entry(
        tmp_path, "if (count != 0) { input += 2; output[0] = input[1]; }"
    )
    assert lines[1] == "if (select_value_count != 0) {"
    assert lines[2:5] == [
        DECLARATION,
        "input_offset += int(2);",
        "output[0] = input.Load(uint((input_offset + 1)));",
    ]
    assert lines.count(DECLARATION) == 1
    _compile(generated, tmp_path)


def test_entry_scope_rebase_before_loop_is_unchanged(tmp_path):
    generated, lines = _entry(
        tmp_path,
        "input += 1; for (uint i = 0; i < count; ++i)"
        " { output[i] = input[0]; input += 2; }",
    )
    assert lines[1:3] == [DECLARATION, "input_offset += int(1);"]
    assert lines.count(DECLARATION) == 1
    assert "input_offset += int(2);" in lines
    _compile(generated, tmp_path)

"""Switch clauses keep their source order and their own jumps through import.

A ``default`` label can sit anywhere among the cases, and a clause without a
jump falls through into the next clause in source order. The HLSL, GLSL and
Metal importers used to move ``default`` after the last case, drop an empty
``default:``, and (Metal) end every clause with an inserted ``break``. Each of
these changes which statements run for some selector values.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.common_ast import CaseNode, SwitchNode
from crosstl.backend.switch_clauses import (
    ordered_switch_clauses,
    record_default_position,
)

SWITCH_BODY = """
    switch (sel) {
        case 4:
        default:
            r += 1;
        case 5:
            r += 10;
            break;
        case 0:
        case 1:
            r = 100;
            break;
        case 2:
            r = 7;
            break;
    }
"""

DEFAULT_FIRST_BODY = """
    switch (sel) {
        default:
            r = 3;
        case 1:
            r += 20;
            break;
        case 2:
            r = 9;
            break;
    }
"""

EMPTY_DEFAULT_BODY = """
    switch (sel) {
        case 1:
            r = 1;
            break;
        default:
        case 2:
            r = 2;
            break;
    }
"""


def _hlsl(body):
    return (
        "RWStructuredBuffer<int> outBuf : register(u0);\n"
        "[numthreads(1, 1, 1)]\n"
        "void main(uint3 id : SV_DispatchThreadID) {\n"
        "    int sel = (int)id.x;\n"
        "    int r = 0;\n"
        f"{body}"
        "    outBuf[id.x] = r;\n"
        "}\n"
    )


def _glsl(body):
    return (
        "#version 450\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer Out { int outBuf[]; };\n"
        "void main() {\n"
        "    int sel = int(gl_GlobalInvocationID.x);\n"
        "    int r = 0;\n"
        f"{body}"
        "    outBuf[gl_GlobalInvocationID.x] = r;\n"
        "}\n"
    )


def _metal(body):
    return (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void k(device int* outBuf [[buffer(0)]],\n"
        "              uint id [[thread_position_in_grid]]) {\n"
        "    int sel = (int)id;\n"
        "    int r = 0;\n"
        f"{body}"
        "    outBuf[id] = r;\n"
        "}\n"
    )


SOURCES = {
    "hlsl": (_hlsl, "hlsl"),
    "glsl": (_glsl, "comp"),
    "metal": (_metal, "metal"),
}


def _import(tmp_path, language, body):
    render, suffix = SOURCES[language]
    path = tmp_path / f"shader.{suffix}"
    path.write_text(render(body))
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def _switch_labels(code):
    """The switch's labels and statements, one token run per line, in order."""
    start = code.index("switch")
    depth = 0
    lines = []
    for line in code[start:].splitlines()[1:]:
        text = line.strip()
        if text.startswith("}") and depth == 0:
            break
        depth += text.count("{") - text.count("}")
        if text:
            lines.append(text)
    return lines


@pytest.mark.parametrize("language", sorted(SOURCES))
def test_default_keeps_its_position_and_fallthrough(tmp_path, language):
    labels = _switch_labels(_import(tmp_path, language, SWITCH_BODY))
    assert labels == [
        "case 4:",
        "default:",
        "r += 1;",
        "case 5:",
        "r += 10;",
        "break;",
        "case 0:",
        "case 1:",
        "r = 100;",
        "break;",
        "case 2:",
        "r = 7;",
        "break;",
    ], labels


@pytest.mark.parametrize("language", sorted(SOURCES))
def test_default_first_falls_through_into_the_next_case(tmp_path, language):
    labels = _switch_labels(_import(tmp_path, language, DEFAULT_FIRST_BODY))
    assert labels == [
        "default:",
        "r = 3;",
        "case 1:",
        "r += 20;",
        "break;",
        "case 2:",
        "r = 9;",
        "break;",
    ], labels


@pytest.mark.parametrize("language", sorted(SOURCES))
def test_empty_default_is_kept_before_the_case_it_shares(tmp_path, language):
    labels = _switch_labels(_import(tmp_path, language, EMPTY_DEFAULT_BODY))
    assert labels == [
        "case 1:",
        "r = 1;",
        "break;",
        "default:",
        "case 2:",
        "r = 2;",
        "break;",
    ], labels


def test_switch_without_a_recorded_position_puts_default_last():
    node = SwitchNode("x", [CaseNode(1, ["a"]), CaseNode(2, ["b"])], ["d"])
    assert ordered_switch_clauses(node) == [(1, ["a"]), (2, ["b"]), (None, ["d"])]
    record_default_position(node, 1)
    assert ordered_switch_clauses(node) == [(1, ["a"]), (None, ["d"]), (2, ["b"])]


def test_switch_without_default_has_only_cases():
    node = record_default_position(SwitchNode("x", [CaseNode(1, ["a"])]), None)
    assert ordered_switch_clauses(node) == [(1, ["a"])]


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_imported_switch_order_reaches_validated_targets(tmp_path, backend):
    """HLSL to GLSL and SPIR-V keep ``case 4`` sharing the default's body."""
    path = tmp_path / "shader.hlsl"
    path.write_text(_hlsl(SWITCH_BODY))
    code = crosstl.translate(str(path), backend=backend, format_output=False)
    if backend == "opengl":
        flat = " ".join(code.split())
        assert "case 4: default: { r += 1; } case 5: { r += 10; break; }" in flat
        validator = shutil.which("glslangValidator")
        if validator is None:
            return
        out = tmp_path / "out.comp"
        out.write_text(code)
        command = [validator, str(out)]
    else:
        # OpSwitch %sel %default 4 %case4 ...: the case 4 block branches straight to the default.
        words = next(line.split() for line in code.splitlines() if "OpSwitch" in line)
        default_label = words[2]
        targets = dict(zip(words[3::2], words[4::2]))
        lines = [line.strip() for line in code.splitlines()]
        case4_block = lines.index(f"{targets['4']} = OpLabel")
        assert lines[case4_block + 1] == f"OpBranch {default_label}", code
        assembler = shutil.which("spirv-as")
        validator = shutil.which("spirv-val")
        if assembler is None or validator is None:
            return
        source = tmp_path / "out.spvasm"
        source.write_text(code)
        binary = tmp_path / "out.spv"
        assembled = subprocess.run(
            [assembler, str(source), "-o", str(binary), "--target-env", "vulkan1.1"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert assembled.returncode == 0, assembled.stderr
        command = [validator, str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr + code

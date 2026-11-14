"""HIP's warpSize and shuffle widths keep their meaning on wavefronts of 32 and 64 lanes.

HIP's ``warpSize`` is the device's wavefront width, 32 or 64 lanes, and a shuffle with an explicit ``width``
works inside ``width``-lane segments of the wavefront (HIP programming guide, "Warp cross-lane functions";
``amd_warp_functions.h``). The importer used to replace ``warpSize`` with 32, which is wrong on every wave64
device. These tests evaluate the imported CrossGL shuffle expressions lane by lane and compare them with HIP's
definition for both wavefront widths.
"""

import re

import pytest

from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser


def _import(code):
    return HipToCrossGLConverter().generate(
        HipParser(HipLexer(code).tokenize()).parse()
    )


def _assigned_expression(crossgl, name):
    match = re.search(rf"var {name}: i32 = (.+);\n", crossgl)
    assert match, crossgl
    return match.group(1)


def _split_top_level(text, separator):
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == separator and depth == 0:
            return text[:index], text[index + 1 :]
    return None


def _python_expression(text):
    """Rewrite one C-style CrossGL expression (calls, arithmetic, ``?:``) as Python."""
    text = text.strip()
    question = _split_top_level(text, "?")
    if question is not None:
        condition, branches = question
        then, otherwise = _split_top_level(branches, ":")
        return (
            f"({_python_expression(then)} if {_python_expression(condition)} "
            f"else {_python_expression(otherwise)})"
        )
    pieces, depth, start = [], 0, None
    for index, character in enumerate(text):
        if character == "(":
            if depth == 0:
                start = index
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                inner = text[start + 1 : index]
                arguments = []
                rest = inner
                while (split := _split_top_level(rest, ",")) is not None:
                    arguments.append(_python_expression(split[0]))
                    rest = split[1]
                arguments.append(_python_expression(rest) if rest.strip() else "")
                pieces.append((start, index, ", ".join(arguments)))
    result, cursor = "", 0
    for start, end, inner in pieces:
        result += text[cursor:start] + "(" + inner + ")"
        cursor = end + 1
    return (result + text[cursor:]).replace("%", " % ").replace("^", " ^ ")


def _evaluate(expression, values, wave):
    python = _python_expression(expression)
    results = []
    for lane in range(wave):

        def read_lane(value, index):
            assert 0 <= index < wave, (python, lane, index)
            return values[index]

        scope = {
            "WaveGetLaneIndex": lambda lane=lane: lane,
            "WaveGetLaneCount": lambda: wave,
            "WaveReadLaneAt": read_lane,
            "gl_SubgroupSize": wave,
            "int": int,
            "value": values[lane],
        }
        results.append(
            eval(python, scope)
        )  # noqa: S307 - expression built by the test from generated text
    return results


def _hip_reference(function, values, operand, width, wave):
    """HIP's __shfl, __shfl_up, __shfl_down and __shfl_xor (amd_warp_functions.h)."""
    results = []
    for lane in range(wave):
        if function == "shfl":
            index = (operand & (width - 1)) + (lane & -width)
        elif function == "up":
            index = lane - operand
            index = lane if index < (lane & -width) else index
        elif function == "down":
            index = lane + operand
            index = lane if (lane & (width - 1)) + operand >= width else index
        else:
            index = lane ^ operand
            index = lane if index >= ((lane + width) & -width) else index
        results.append(values[index])
    return results


SHUFFLES = {
    "shfl": "__shfl_sync(0xffffffffffffffff, value, {operand}{width})",
    "up": "__shfl_up_sync(0xffffffffffffffff, value, {operand}{width})",
    "down": "__shfl_down_sync(0xffffffffffffffff, value, {operand}{width})",
    "xor": "__shfl_xor_sync(0xffffffffffffffff, value, {operand}{width})",
}


@pytest.mark.parametrize("function", sorted(SHUFFLES))
@pytest.mark.parametrize("width", ["", "warpSize", "32", "16"])
@pytest.mark.parametrize("wave", [32, 64])
def test_shuffle_widths_follow_hip_segments(function, width, wave):
    operands = {"shfl": 37, "up": 3, "down": 5, "xor": 4}
    operand = operands[function]
    call = SHUFFLES[function].format(
        operand=operand, width=f", {width}" if width else ""
    )
    crossgl = _import(
        f"__global__ void probe(int* out, int value) {{\n  int r = {call};\n  out[threadIdx.x] = r;\n}}\n"
    )
    assert "warpSize" not in crossgl
    expression = _assigned_expression(crossgl, "r")
    values = [100 + lane for lane in range(wave)]
    segment = wave if width in {"", "warpSize"} else int(width)
    if function == "shfl" and segment == wave:
        operand %= wave  # a full-width read names a lane of the wavefront
        expression = expression.replace("37", str(operand))
    assert _evaluate(expression, values, wave) == _hip_reference(
        function, values, operand, segment, wave
    )


def test_warp_size_is_the_subgroup_size_not_32():
    crossgl = _import(
        "__global__ void probe(int* out) {\n"
        "  int lanes = warpSize;\n"
        "  int lane = (int)threadIdx.x % warpSize;\n"
        "  out[threadIdx.x] = lanes + lane;\n"
        "}\n"
    )
    assert "var lanes: i32 = int(gl_SubgroupSize);" in crossgl
    assert "% int(gl_SubgroupSize))" in crossgl
    body = crossgl[crossgl.index(") {") :]
    assert re.search(r"\b32\b", body) is None

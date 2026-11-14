"""A CrossGL switch case without a jump falls through into the next case in the generated SPIR-V.

SPIR-V expresses fall-through as a branch from one case construct to the next case's block, and requires a case that
falls through to precede the case it enters in the ``OpSwitch`` target list (SPIR-V 1.6 section 2.11.1). The code
generator used to branch every case to the merge block, which silently dropped the statements of the following cases.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen

SOURCE = """
shader SwitchFallThrough {
    compute {
        @stage_entry
        void main(
            RWStructuredBuffer<uint> words @binding(0),
            uint gid @gl_GlobalInvocationID
        ) {
            uint value = 0u;
            switch (int(gid & 7u)) {
                case 0:
                    value = value + 1u;
                case 1:
                    value = value + 10u;
                    break;
                default:
                    value = value + 100u;
                case 5:
                    value = value + 1000u;
                    break;
                case 6:
                    value = value + 10000u;
            }
            buffer_store(words, gid, value);
        }
    }
}
"""


def _labels(assembly):
    """The ``OpSwitch`` default and case labels, and the branch that ends each block."""
    switch = re.search(r"OpSwitch %\w+ %(\w+)((?: -?\d+ %\w+)*)", assembly)
    assert switch, assembly
    cases = dict(re.findall(r"(-?\d+) %(\w+)", switch.group(2)))
    exits = {}
    label = None
    for line in assembly.splitlines():
        line = line.strip()
        opened = re.match(r"%(\w+) = OpLabel", line)
        if opened:
            label = opened.group(1)
        elif line.startswith("OpBranch ") and label is not None:
            exits[label] = line.split("%", 1)[1]
            label = None
    return switch.group(1), cases, exits


def _generate():
    return VulkanSPIRVCodeGen().generate(crosstl.translator.parse(SOURCE))


def test_cases_without_a_jump_branch_to_the_next_case():
    default, cases, exits = _labels(_generate())
    # case 0 enters case 1; the default (between case 1 and case 5 in source order) enters case 5.
    assert exits[cases["0"]] == cases["1"]
    assert exits[default] == cases["5"]
    # case 1 and case 5 break, and the last case ends the switch, so all three reach one merge block.
    assert exits[cases["1"]] == exits[cases["5"]] == exits[cases["6"]]
    assert exits[cases["1"]] not in set(cases.values()) | {default}


def test_fall_through_module_validates(tmp_path):
    assembly = _generate()
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    source = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr

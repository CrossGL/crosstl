"""SPIR-V keeps the short-circuit evaluation of ``&&``, ``||`` and ``?:``.

GLSL, MSL, C++ and HLSL 2021 run the right operand of a scalar ``&&``/``||`` only when the left
one leaves the result open, and only the selected arm of ``?:``. The tests check that an operand
which writes or can fault runs in its own block of a selection construct whose merge block joins
the values with ``OpPhi``, that a loop condition which branches gets its own block behind the loop
header, and that operands which can neither write nor fault keep the single-instruction form.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_short_circuit import expression_is_trivial
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def spirv_for(body: str, helpers: str = "") -> str:
    source = f"""
    shader ShortCircuit {{
        RWStructuredBuffer<int> data;
        {helpers}
        compute {{
            layout(local_size_x = 1) in;
            void main() {{
                int i = int(gl_GlobalInvocationID.x);
                int n = 4;
                int counter = 0;
                {body}
            }}
        }}
    }}
    """
    return VulkanSPIRVCodeGen().generate(Parser(Lexer(source).tokens).parse())


BUMP = """
    bool bump(inout int counter) {
        counter = counter + 1;
        return counter > 2;
    }
"""


def validate(spv_code: str, tmp_path) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm_path = tmp_path / "shader.spvasm"
    spv_path = tmp_path / "shader.spv"
    asm_path.write_text(spv_code)
    assemble = subprocess.run(
        [spirv_as, str(asm_path), "-o", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assemble.returncode == 0, assemble.stderr
    result = subprocess.run(
        [spirv_val, str(spv_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def main_body(spv_code: str) -> list:
    """Return the instructions of the entry point, the last function in the module."""
    lines = spv_code.splitlines()
    start = max(index for index, line in enumerate(lines) if "OpFunction " in line)
    return lines[start:]


def blocks(lines: list) -> dict:
    """Map each label of ``lines`` to the instructions of its block."""
    result, label = {}, None
    for line in lines:
        match = re.match(r"(%\d+) = OpLabel$", line)
        if match:
            label = match.group(1)
            result[label] = []
        elif label is not None:
            result[label].append(line)
    return result


def block_of(lines: list, pattern: str) -> str:
    """Return the label of the only block holding an instruction that matches ``pattern``."""
    found = [
        label
        for label, body in blocks(lines).items()
        if any(re.search(pattern, line) for line in body)
    ]
    assert len(found) == 1, (pattern, found)
    return found[0]


def counter_store_blocks(spv_code: str, lines: list) -> set:
    """Return the labels of the blocks after the entry block that store to ``counter``."""
    # the helper's parameter is named `counter` too; the entry point's variable comes last
    counter = re.findall(r'OpName (%\d+) "counter"', spv_code)[-1]
    labelled = blocks(lines)
    entry = next(iter(labelled))
    return {
        label
        for label, body in labelled.items()
        if label != entry
        and any(line.startswith(f"OpStore {counter} ") for line in body)
    }


def phi(lines: list) -> re.Match:
    phis = [
        re.match(r"(%\d+) = OpPhi (%\d+) (%\d+) (%\d+) (%\d+) (%\d+)$", line)
        for line in lines
        if " = OpPhi " in line
    ]
    assert len(phis) == 1, phis
    assert phis[0] is not None
    return phis[0]


def test_logical_and_reads_the_buffer_only_when_the_bound_check_holds(tmp_path):
    spv_code = spirv_for("bool ok = i < n && data[i] > 0; data[0] = ok ? 1 : 0;")
    validate(spv_code, tmp_path)
    lines = main_body(spv_code)
    assert not any("OpLogicalAnd" in line for line in lines)

    branch = next(line for line in lines if line.startswith("OpBranchConditional"))
    _, check, guarded, merge = branch.split()
    assert block_of(lines, r"OpSGreaterThan ") == guarded
    assert block_of(lines, r"OpSLessThan ") != guarded
    merged = phi(lines)
    # false from the check's block, the guarded comparison from the guarded block
    assert merged.group(3) == check and merged.group(5) != check
    assert merged.group(6) == guarded
    assert f"{merge} = OpLabel" in lines


def test_logical_or_runs_a_writing_call_only_when_the_left_operand_is_false(tmp_path):
    spv_code = spirv_for(
        "bool seen = data[1] == 7; bool any = seen || bump(counter);"
        " data[0] = counter + (any ? 10 : 0);",
        BUMP,
    )
    validate(spv_code, tmp_path)
    lines = main_body(spv_code)
    assert not any("OpLogicalOr" in line for line in lines)

    branch = next(line for line in lines if line.startswith("OpBranchConditional"))
    _, seen, merge, guarded = branch.split()
    merged = phi(lines)
    assert merged.group(3) == seen
    assert merged.group(6) == guarded
    assert f"{merge} = OpLabel" in lines
    # the inlined call's increment of `counter` happens only in the guarded block
    assert counter_store_blocks(spv_code, lines) == {guarded}


def test_division_guarded_by_a_zero_test_runs_only_when_the_divisor_is_nonzero(
    tmp_path,
):
    spv_code = spirv_for(
        "int d = data[1]; bool big = d != 0 && n / d > 1; data[0] = big ? 1 : 0;"
    )
    validate(spv_code, tmp_path)
    lines = main_body(spv_code)
    branch = next(line for line in lines if line.startswith("OpBranchConditional"))
    assert block_of(lines, r"= OpSDiv ") == branch.split()[2]


def test_conditional_expression_runs_only_the_selected_arm(tmp_path):
    spv_code = spirv_for(
        "int value = i < n ? data[i] : (bump(counter) ? 1 : 2); data[0] = value + counter;",
        BUMP,
    )
    validate(spv_code, tmp_path)
    lines = main_body(spv_code)
    branch = next(line for line in lines if line.startswith("OpBranchConditional"))
    _, _, read_arm, call_arm = branch.split()
    assert counter_store_blocks(spv_code, lines) == {call_arm}
    loads_in_read_arm = [line for line in blocks(lines)[read_arm] if "OpLoad" in line]
    assert loads_in_read_arm
    merged = phi(lines)
    assert merged.group(4) == read_arm
    # the call arm's inner `?:` of two literals stays a select inside the call arm
    assert block_of(lines, r"= OpSelect ") == merged.group(6)


def test_conditional_expression_converts_each_arm_inside_its_own_block(tmp_path):
    spv_code = spirv_for(
        "float value = i < n ? data[i] : 2.5; data[0] = int(value);",
    )
    validate(spv_code, tmp_path)
    lines = main_body(spv_code)
    branch = next(line for line in lines if line.startswith("OpBranchConditional"))
    read_arm = branch.split()[2]
    converted = block_of(lines, r"= OpConvertSToF ")
    assert converted == read_arm
    float_type = re.search(r"(%\d+) = OpTypeFloat 32\b", spv_code).group(1)
    assert phi(lines).group(2) == float_type


def test_while_condition_that_branches_gets_a_block_after_the_loop_header(tmp_path):
    spv_code = spirv_for(
        "while (i < n && data[i] != 0) { i = i + 1; } data[0] = i;",
    )
    validate(spv_code, tmp_path)
    lines = main_body(spv_code)
    merge_index = next(i for i, line in enumerate(lines) if "OpLoopMerge" in line)
    header = lines[merge_index - 1]
    assert re.fullmatch(r"%\d+ = OpLabel", header)
    condition_branch = lines[merge_index + 1]
    condition_label = condition_branch.split()[1]
    assert condition_branch == f"OpBranch {condition_label}"
    assert lines[merge_index + 2] == f"{condition_label} = OpLabel"
    merged = phi(lines)
    assert merged.group(4) == condition_label
    assert header.split()[0] not in merged.group(0).split()[3:]


def test_for_condition_with_an_inlined_branching_call_validates(tmp_path):
    helpers = """
        bool below(int value, int limit) {
            if (value < 0) {
                return false;
            }
            return value < limit;
        }
    """
    spv_code = spirv_for(
        "int sum = 0; for (int k = 0; below(k, n) || bump(counter); k++) {"
        " sum = sum + k; if (k > 8) { break; } } data[0] = sum + counter;",
        BUMP + helpers,
    )
    validate(spv_code, tmp_path)


def test_do_while_condition_with_a_guarded_operand_validates(tmp_path):
    spv_code = spirv_for(
        "do { i = i + 1; } while (i < n && data[i] > 0); data[0] = i;",
    )
    validate(spv_code, tmp_path)
    assert "OpPhi" in spv_code


def test_nested_guards_in_an_if_condition_validate(tmp_path):
    spv_code = spirv_for(
        "if (i < n && (data[i] > 0 || bump(counter))) { data[0] = counter; }",
        BUMP,
    )
    validate(spv_code, tmp_path)
    assert len(re.findall(r"= OpPhi ", spv_code)) == 2


def test_operands_that_cannot_write_or_fault_keep_single_instructions(tmp_path):
    spv_code = spirv_for(
        "bool a = i < n && counter == 0; bool b = a || -i > n;"
        " int c = a ? i : n + 1; data[0] = c + (b ? 1 : 0);",
    )
    validate(spv_code, tmp_path)
    lines = main_body(spv_code)
    assert any("OpLogicalAnd" in line for line in lines)
    assert any("OpLogicalOr" in line for line in lines)
    assert not any("OpPhi" in line for line in lines)
    assert not any(line.startswith("OpBranchConditional") for line in lines)


@pytest.mark.parametrize(
    "expression, trivial",
    [
        ("a < b && !c", True),
        ("float(i) * 2.0 + x.y", True),
        ("a / 2", True),
        ("a / b", False),
        ("a % 0", False),
        ("values[i]", False),
        ("f(a)", False),
        ("i++", False),
        ("c ? a : b", True),
    ],
)
def test_trivial_operands_exclude_writes_reads_through_indices_and_division(
    expression, trivial
):
    source = f"shader T {{ compute {{ void main() {{ bool r = {expression}; }} }} }}"
    ast = Parser(Lexer(source).tokens).parse()
    (stage,) = ast.stages.values()
    initializer = stage.entry_point.body.statements[0].initial_value
    assert expression_is_trivial(initializer) is trivial

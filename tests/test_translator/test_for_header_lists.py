"""``for`` headers with several initializers or updates, updates with side effects, and ``do``-``while`` loops.

``for (k = 0, m = 3; ...)`` did not parse: the CrossGL and GLSL parsers read one expression before the first ``;``.
WGSL has no comma operator and no ``do``-``while``, and its ``for`` header holds one initializer and one update, so
the WGSL emitter raised an untyped ``ValueError`` for ``i++, j--``, ``data[q++] = z`` and every ``do``-``while``.
The SPIR-V emitter silently dropped a list of updates, so ``for (...; i < j; i++, j--)`` never terminated.
The Mojo emitter printed the Python repr of a list of initializers or updates.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.ast import (
    AssignmentNode,
    ASTNode,
    ForNode,
    UnaryOpNode,
    VariableNode,
)
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.rewrite import child_slots, get_child
from tests.spirv_friendly_names import translate_spirv

COUNTERS = """
int total = 0;
for (int i = 0, j = 10; i < j; i++, j--) {
    if (i == 1) {
        continue;
    }
    total += i * j;
}
int k;
int m;
for (k = 0, m = 3; k < m; ++k, m -= 1) {
    total += k;
}
int q = 0;
for (int z = 0; z < 4; data[q++] = z, z++) {
}
int c = 3;
do {
    total += c;
    if (c == 2) {
        continue;
    }
} while (--c > 0);
do total += 1; while (false);
data[0] = total;
"""


def compute_source(body):
    return textwrap.dedent("""
        shader Probe {
            compute {
                buffer int data[];
                [numthreads(1, 1, 1)]
                void main() {
        %s
                }
            }
        }
        """) % textwrap.indent(textwrap.dedent(body), " " * 12)


def parse(body):
    return Parser(Lexer(compute_source(body)).get_tokens()).parse()


def loops(node, found=None):
    """Every ``ForNode`` below ``node``, in source order."""
    found = [] if found is None else found
    if isinstance(node, ForNode):
        found.append(node)
    for field, key in child_slots(node):
        loops(get_child(node, field, key), found)
    return found


def body_lines(output):
    entry = output[output.index("fn compute_main") :]
    return [line.strip() for line in entry.splitlines()[1:] if line.strip()]


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def naga_validate(tmp_path, wgsl):
    if shutil.which("naga") is None:
        pytest.skip("naga is not installed")
    path = write(tmp_path, "probe.wgsl", wgsl)
    result = subprocess.run(["naga", path], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_assignment_initializers_and_updates_parse_as_lists():
    (loop,) = loops(parse("int k;\nint m;\nfor (k = 0, m = 3; k < m; ++k, m -= 1) { }"))
    assert [type(item) for item in loop.init] == [AssignmentNode, AssignmentNode]
    assert [item.target.name for item in loop.init] == ["k", "m"]
    assert [type(item) for item in loop.update] == [UnaryOpNode, AssignmentNode]


def test_single_initializer_stays_one_statement():
    (loop,) = loops(parse("int k;\nfor (k = 0; k < 3; ++k) { }"))
    assert not isinstance(loop.init, list)


def test_glsl_import_keeps_assignment_initializer_lists(tmp_path):
    source = textwrap.dedent("""
        #version 450
        layout(local_size_x = 1) in;
        layout(std430, binding = 0) buffer Outp { int o[]; } outp;
        void main() {
            int k;
            int m;
            for (k = 0, m = 3; k < m; ++k, m -= 1) {
                outp.o[k] = m;
            }
        }
        """)
    crossgl = crosstl.translate(write(tmp_path, "loop.comp", source), backend="cgl")
    assert re.search(r"for \(k = 0, m = 3; \(k < m\); \(\+\+k\), m -= 1\)", crossgl)
    (loop,) = loops(Parser(Lexer(crossgl).get_tokens()).parse())
    assert [item.target.name for item in loop.init] == ["k", "m"]


def test_wgsl_runs_several_updates_in_a_continuing_block():
    lines = body_lines(WGSLCodeGen().generate(parse(COUNTERS)))
    start = lines.index("var i: i32 = 0;")
    assert lines[start : start + 4] == [
        "var i: i32 = 0;",
        "var j: i32 = 10;",
        "loop {",
        "if (!(i < j))             {",
    ]
    continuing = lines.index("continuing {", start)
    assert lines[continuing + 1 : continuing + 3] == ["i += 1;", "j -= 1;"]
    # ``continue`` stays inside the loop, so it reaches the continuing block.
    assert start < lines.index("continue;") < continuing


def test_wgsl_runs_assignment_initializers_before_the_loop():
    lines = body_lines(WGSLCodeGen().generate(parse(COUNTERS)))
    start = lines.index("k = 0;")
    assert lines[start + 1] == "m = 3;"
    assert lines[start + 2] == "loop {"
    continuing = lines.index("continuing {", start)
    assert lines[continuing + 1 : continuing + 3] == ["k += 1;", "m -= 1;"]


def test_wgsl_update_that_stores_through_an_increment_runs_in_order():
    lines = body_lines(WGSLCodeGen().generate(parse(COUNTERS)))
    start = lines.index("var z: i32 = 0;")
    continuing = lines.index("continuing {", start)
    assert lines[continuing + 1 : continuing + 5] == [
        "let _crosstl_prior_0: i32 = q;",
        "q += 1;",
        "data[_crosstl_prior_0] = z;",
        "z += 1;",
    ]


def test_wgsl_do_while_tests_its_condition_after_continue():
    lines = body_lines(WGSLCodeGen().generate(parse(COUNTERS)))
    start = lines.index("var c: i32 = 3;")
    assert lines[start + 1] == "loop {"
    continuing = lines.index("continuing {", start)
    assert start < lines.index("continue;", start) < continuing
    assert lines[continuing + 1 : continuing + 4] == [
        "c -= 1;",
        "break if !(c > 0);",
        "}",
    ]
    assert "break if !false;" in lines
    assert not any(line.startswith("do") for line in lines)


def test_wgsl_loops_validate(tmp_path):
    naga_validate(tmp_path, WGSLCodeGen().generate(parse(COUNTERS)))


def continue_block(spirv, counter_names):
    """The instructions of the continue target of the loop whose condition reads ``counter_names``."""
    lines = spirv.splitlines()
    for index, line in enumerate(lines):
        merge = re.search(r"OpLoopMerge %\w+ (%\w+)", line)
        if not merge:
            continue
        header = "\n".join(lines[max(0, index - 4) : index])
        if not all(f"OpLoad %int %{name}" in header for name in counter_names):
            continue
        label = lines.index(
            next(
                text for text in lines if text.strip() == f"{merge.group(1)} = OpLabel"
            )
        )
        block = []
        for text in lines[label + 1 :]:
            block.append(text.strip())
            if text.strip().startswith("OpBranch"):
                return block
    raise AssertionError(f"no loop over {counter_names}")


def test_spirv_runs_every_update_of_the_list(tmp_path):
    spirv = translate_spirv(write(tmp_path, "loop.cgl", compute_source(COUNTERS)))
    for counters in (("i", "j"), ("k", "m")):
        stores = [
            line
            for line in continue_block(spirv, counters)
            if line.startswith("OpStore")
        ]
        assert [line.split()[1] for line in stores] == [f"%{name}" for name in counters]


def test_spirv_runs_assignment_initializers_before_the_loop(tmp_path):
    spirv = translate_spirv(write(tmp_path, "loop.cgl", compute_source(COUNTERS)))
    lines = [line.strip() for line in spirv.splitlines()]
    start = lines.index("OpStore %k %int_0")
    assert lines[start + 1] == "OpStore %m %int_3"


def test_spirv_refuses_a_statement_it_cannot_lower():
    class UnknownStatement(ASTNode):
        pass

    codegen = VulkanSPIRVCodeGen()
    codegen.process_statement(UnknownStatement())
    assert codegen.refusals.reasons == [
        "statement UnknownStatement has no SPIR-V lowering"
    ]


def test_mojo_runs_initializer_and_update_lists_in_order(tmp_path):
    body = """
    int total = 0;
    for (int i = 0, j = 10; i < j; i++, j--) {
        total += i * j;
    }
    int k;
    int m;
    for (k = 0, m = 3; k < m; ++k, m -= 1) {
        total += k;
    }
    data[0] = total;
    """
    mojo = crosstl.translate(
        write(tmp_path, "loop.cgl", compute_source(body)), backend="mojo"
    )
    assert "Node(" not in mojo
    lines = [line.strip() for line in mojo.splitlines()]
    first = lines.index("var i: Int32 = 0")
    assert lines[first : first + 6] == [
        "var i: Int32 = 0",
        "var j: Int32 = 10",
        "while (i < j):",
        "total += (i * j)",
        "i += 1",
        "j -= 1",
    ]
    second = lines.index("k = 0")
    assert lines[second : second + 6] == [
        "k = 0",
        "m = 3",
        "while (k < m):",
        "total += k",
        "k += 1",
        "m -= 1",
    ]


def test_for_declaration_list_keeps_variable_nodes():
    (loop,) = loops(parse("for (int i = 0, j = 4; i < j; i++, j--) { }"))
    assert [type(item) for item in loop.init] == [VariableNode, VariableNode]

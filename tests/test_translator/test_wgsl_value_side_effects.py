"""WGSL lowering of increments, decrements and assignments used as values.

WGSL has only increment, decrement and assignment statements (WGSL 9.2, 9.3).
The emitter used to print ``a[i++]`` as ``a[i += 1]`` and ``x = y = 7`` as one
expression, which naga rejects, and ``int a = i++`` read the incremented value.
"""

import shutil
import subprocess
import textwrap

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def compute(body, helpers=""):
    source = textwrap.dedent("""
        shader Probe {
            compute {
                buffer int data[];
        %s
                [numthreads(1, 1, 1)]
                void main() {
        %s
                }
            }
        }
        """) % (textwrap.indent(helpers, " " * 8), textwrap.indent(body, " " * 12))
    return WGSLCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def body_lines(output):
    """The stripped statement lines of the entry point."""
    entry = output[output.index("fn compute_main") :]
    return [line.strip() for line in entry.splitlines()[1:] if line.strip()]


def follows(lines, *expected):
    """True when ``expected`` occurs as consecutive lines."""
    count = len(expected)
    return any(
        tuple(lines[i : i + count]) == expected for i in range(len(lines) - count + 1)
    )


def test_postfix_increment_value_reads_the_prior_value():
    lines = body_lines(compute("int i = 0;\nint a = i++;\ndata[0] = a;"))
    assert follows(
        lines,
        "let _crosstl_prior_0: i32 = i;",
        "i += 1;",
        "var a: i32 = _crosstl_prior_0;",
    )


def test_prefix_decrement_value_reads_the_updated_value():
    lines = body_lines(compute("int j = 5;\nint c = -(--j);\ndata[0] = c;"))
    assert follows(lines, "j -= 1;", "var c: i32 = -j;")


def test_increment_in_an_index_runs_before_the_store():
    lines = body_lines(compute("int i = 0;\ndata[i++] = 7;"))
    assert follows(
        lines,
        "let _crosstl_prior_0: i32 = i;",
        "i += 1;",
        "data[_crosstl_prior_0] = 7;",
    )


def test_increment_of_an_indexed_element_lowers_the_index_first():
    lines = body_lines(
        compute(
            "int arr[3];\nint idx = 0;\nint before = arr[idx++]++;\ndata[0] = before;"
        )
    )
    assert follows(
        lines,
        "let _crosstl_prior_0: i32 = idx;",
        "idx += 1;",
        "let _crosstl_prior_1: i32 = arr[_crosstl_prior_0];",
        "arr[_crosstl_prior_0] += 1;",
        "var before: i32 = _crosstl_prior_1;",
    )


def test_chained_and_operand_assignments_become_statements():
    lines = body_lines(
        compute(
            "int x = 0;\nint y = 0;\nx = y = 7;\nint f = (x = 4) + 1;\ndata[0] = f;"
        )
    )
    assert follows(lines, "y = 7;", "x = y;")
    assert follows(lines, "x = 4;", "var f: i32 = (x + 1);")


def test_call_argument_and_return_values_are_lowered():
    output = compute(
        "int idx = 0;\ndata[0] = twice(idx++);",
        helpers="int twice(int v) { return v * 2; }\n"
        "int next() { int c = 0; return ++c; }",
    )
    lines = body_lines(output)
    assert follows(
        lines,
        "let _crosstl_prior_0: i32 = idx;",
        "idx += 1;",
        "data[0] = twice(_crosstl_prior_0);",
    )
    assert "c += 1;\n    return c;" in output


def test_while_condition_side_effects_run_on_every_test():
    lines = body_lines(
        compute("int n = 3;\nint loops = 0;\nwhile (n-- > 0) { loops += 1; }")
    )
    assert "while" not in " ".join(lines)
    assert follows(
        lines,
        "loop     {",
        "let _crosstl_prior_0: i32 = n;",
        "n -= 1;",
        "if (!(_crosstl_prior_0 > 0))         {",
        "break;",
    )


def test_for_condition_side_effects_move_into_the_body_guard():
    lines = body_lines(
        compute("int acc = 0;\nfor (int m = 0; m++ < 4; m += 1) { acc += m; }")
    )
    assert follows(
        lines,
        "for (var m: i32 = 0; ; m += 1)     {",
        "let _crosstl_prior_0: i32 = m;",
        "m += 1;",
        "if (!(_crosstl_prior_0 < 4))         {",
        "break;",
    )


def test_else_if_condition_side_effects_run_only_in_the_else_branch():
    lines = body_lines(
        compute(
            "int k = 0;\nint s = 0;\n"
            "if (k == 1) { s = 1; } else if (++k == 1) { s = 2; } else { s = 3; }"
        )
    )
    assert follows(lines, "} else     {", "k += 1;", "if ((k == 1))         {")
    assert lines.index("k += 1;") > lines.index("s = 1;")


def test_statement_level_updates_are_unchanged():
    lines = body_lines(
        compute("int i = 0;\ni++;\n--i;\nfor (int m = 0; m < 4; m++) { i += m; }")
    )
    assert follows(lines, "i += 1;", "i -= 1;")
    assert "for (var m: i32 = 0; (m < 4); m += 1)     {" in lines
    assert not any("_crosstl_prior_" in line for line in lines)


@pytest.mark.parametrize(
    "body, message",
    [
        ("int i = 0;\nbool b = (i > 0) && (i++ > 1);", "right operand of && or ||"),
        ("int i = 0;\nint t = (i > 0) ? i++ : 0;", "ternary branch"),
        ("int i = 0;\nfor (int m = 0; m < 4; data[i++] = m) { }", "for-loop update"),
        ("int arr[2];\nint v = arr[pick()]++;", "evaluate twice"),
    ],
)
def test_unorderable_side_effects_fail_closed(body, message):
    helpers = "int pick() { return 1; }"
    with pytest.raises(ValueError, match=message):
        compute(body, helpers=helpers)


VALIDATION_PROGRAM = """
int n = 3;
int loops = 0;
while (n-- > 0) { loops += 10; }
int acc = 0;
for (int m = 0; m++ < 4; m += 1) { acc += m; }
int k = 0;
int s = 0;
if (k == 1) { s = 1; } else if (++k == 1) { s = 2; } else { s = 3; }
int x = 0;
int y = 0;
x = y = 7;
int arr[3];
arr[0] = 1;
int idx = 0;
int before = arr[idx++]++;
ivec2 v = ivec2(5, 6);
int sw = v.y--;
data[k++] += n++;
data[0] = loops + acc + s + x + y + before + idx + sw;
"""


def test_lowered_program_validates_with_naga(tmp_path):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "side_effects.wgsl"
    path.write_text(compute(VALIDATION_PROGRAM), "utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr

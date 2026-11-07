"""WGSL copy-in/copy-out for ``inout``/``out`` arguments that are not whole local variables.

Core WGSL passes a ``ptr<function, T>`` argument only for a whole function-scope variable (WGSL 11.3.1). The source
languages pass any l-value by copy-in/copy-out (GLSL 4.60 section 6.1.1), so an array element, a member, a buffer
element or a module-scope variable is copied into a local around the call and stored back after it.
"""

import shutil
import subprocess
import textwrap

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

HELPERS = """
int bump(inout int a) { int old = a; a = a + 1; return old * 10; }
void put(out int a, int v) { a = v; }
void inc2(inout ivec2 p) { p = p + ivec2(1, 2); }
int step2(inout int idx, inout int slot) { idx = idx + 1; slot = slot + 100; return idx; }
int poke(int k) { tile[1] = tile[1] + 1000; return k; }
int pair(int k, inout int t) { t = t + k; return t; }
bool below(inout int a, int limit) { a = a + 1; return a < limit; }
int resetTile(inout int t) { tile[2] = 77; t = t + 1; return t; }
int g() { outBuf.values[5] = 1; return 2; }
"""


def compute(body, helpers=HELPERS):
    source = textwrap.dedent("""
        shader Probe {
            struct Counter { int count; ivec2 pair; };
            compute {
                layout(std430, binding = 0) buffer Out { int values[]; } outBuf;
                layout(std430, binding = 1) readonly buffer In { int data[]; } inBuf;
                uniform int scale;
                const int K = 3;
                shared int tile[4];
                layout(local_size_x = 1) in;
        %s
                void main() {
        %s
                }
            }
        }
        """) % (textwrap.indent(helpers, " " * 8), textwrap.indent(body, " " * 12))
    return WGSLCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def main_lines(output):
    start = output.index("fn compute_main(")
    lines = []
    for line in output[start:].splitlines()[1:]:
        if line == "}":
            break
        lines.append(line.strip())
    return lines


def test_array_element_is_copied_in_and_out_around_the_call():
    lines = main_lines(compute("int arr[2];\nint q = 1;\nint z = bump(arr[q]);"))
    assert lines[-4:] == [
        "var _crosstl_ref_0: i32 = arr[q];",
        "let _crosstl_call_1: i32 = bump(&_crosstl_ref_0);",
        "arr[q] = _crosstl_ref_0;",
        "var z: i32 = _crosstl_call_1;",
    ]


def test_member_buffer_element_and_workgroup_variable_are_copied():
    body = "Counter c;\nbump(c.count);\nbump(outBuf.values[3]);\nbump(tile[0]);"
    lines = main_lines(compute(body))
    assert lines[1:] == [
        "var _crosstl_ref_0: i32 = c.count;",
        "bump(&_crosstl_ref_0);",
        "c.count = _crosstl_ref_0;",
        "var _crosstl_ref_1: i32 = outBuf.values[3];",
        "bump(&_crosstl_ref_1);",
        "outBuf.values[3] = _crosstl_ref_1;",
        "var _crosstl_ref_2: i32 = tile[0];",
        "bump(&_crosstl_ref_2);",
        "tile[0] = _crosstl_ref_2;",
    ]


def test_out_argument_starts_from_an_uninitialized_local():
    lines = main_lines(compute("int arr[2];\nput(arr[1], 9);"))
    assert lines[1:] == [
        "var _crosstl_ref_0: i32;",
        "put(&_crosstl_ref_0, 9);",
        "arr[1] = _crosstl_ref_0;",
    ]


def test_multi_component_swizzle_is_stored_back_per_component():
    lines = main_lines(compute("ivec3 v = ivec3(1, 2, 3);\ninc2(v.xz);"))
    assert lines[1:3] == [
        "var _crosstl_ref_0: vec2<i32> = v.xz;",
        "inc2(&_crosstl_ref_0);",
    ]
    assert lines[-2:] == ["v.x = (v_xz).x;", "v.z = (v_xz).y;"]


def test_index_written_by_the_same_call_is_bound_before_copy_in():
    # GLSL evaluates the l-value once, when the call starts: arr[q] is arr[0] although step2 increments q.
    lines = main_lines(compute("int arr[2];\nint q = 0;\nint r = step2(q, arr[q]);"))
    assert lines[2:] == [
        "let _crosstl_arg_0: i32 = q;",
        "var _crosstl_ref_1: i32 = arr[_crosstl_arg_0];",
        "let _crosstl_call_2: i32 = step2(&q, &_crosstl_ref_1);",
        "arr[_crosstl_arg_0] = _crosstl_ref_1;",
        "var r: i32 = _crosstl_call_2;",
    ]


def test_earlier_argument_with_a_call_runs_before_the_copy_in():
    lines = main_lines(compute("int r = pair(poke(5), tile[1]);"))
    assert lines == [
        "let _crosstl_arg_0: i32 = poke(5);",
        "var _crosstl_ref_1: i32 = tile[1];",
        "let _crosstl_call_2: i32 = pair(_crosstl_arg_0, &_crosstl_ref_1);",
        "tile[1] = _crosstl_ref_1;",
        "var r: i32 = _crosstl_call_2;",
    ]


def test_two_copied_calls_in_one_expression_keep_left_to_right_order():
    lines = main_lines(compute("int arr[3];\nint r = bump(arr[2]) + bump(arr[1]);"))
    assert lines[1:] == [
        "var _crosstl_ref_0: i32 = arr[2];",
        "let _crosstl_call_1: i32 = bump(&_crosstl_ref_0);",
        "arr[2] = _crosstl_ref_0;",
        "var _crosstl_ref_2: i32 = arr[1];",
        "let _crosstl_call_3: i32 = bump(&_crosstl_ref_2);",
        "arr[1] = _crosstl_ref_2;",
        "var r: i32 = (_crosstl_call_1 + _crosstl_call_3);",
    ]


def test_while_condition_copies_run_once_per_test():
    lines = main_lines(
        compute("int arr[2];\nint n = 0;\nwhile (below(arr[0], 3)) { n = n + 1; }")
    )
    loop = lines.index("loop     {")
    assert lines[loop + 1 : loop + 5] == [
        "var _crosstl_ref_0: i32 = arr[0];",
        "let _crosstl_call_1: bool = below(&_crosstl_ref_0, 3);",
        "arr[0] = _crosstl_ref_0;",
        "if (!_crosstl_call_1)         {",
    ]


def test_if_and_return_positions_are_copied():
    helpers = HELPERS + "int viaReturn(int i) { int arr[2]; return bump(arr[i]); }\n"
    output = compute(
        "int arr[2];\nif (bump(arr[0]) > 0) { arr[1] = viaReturn(1); }", helpers
    )
    lines = main_lines(output)
    assert lines[1:4] == [
        "var _crosstl_ref_2: i32 = arr[0];",
        "let _crosstl_call_3: i32 = bump(&_crosstl_ref_2);",
        "arr[0] = _crosstl_ref_2;",
    ]
    assert lines[4].startswith("if ((_crosstl_call_3 > 0))")
    assert "return _crosstl_call_1;" in output


def test_ternary_branch_copies_run_only_in_their_branch():
    # The conditional becomes if/else first, so the branch holds its copies.
    lines = main_lines(compute("int arr[2];\nint r = arr[0] > 0 ? bump(arr[1]) : 0;"))
    assert lines[2:9] == [
        "if ((arr[0] > 0))     {",
        "var _crosstl_ref_0: i32 = arr[1];",
        "let _crosstl_call_1: i32 = bump(&_crosstl_ref_0);",
        "arr[1] = _crosstl_ref_0;",
        "r = _crosstl_call_1;",
        "} else     {",
        "r = 0;",
    ]


def test_do_while_condition_copies_run_in_the_continuing_block():
    # continuing runs after every iteration and every continue, as the condition does.
    body = (
        "int arr[2];\nint r = 0;\n"
        "do { r = r + 1; if (r > 5) { continue; } } while (bump(arr[0]) < 3);"
    )
    lines = main_lines(compute(body))
    start = lines.index("continuing {")
    assert lines[start + 1 : start + 6] == [
        "var _crosstl_ref_0: i32 = arr[0];",
        "let _crosstl_call_1: i32 = bump(&_crosstl_ref_0);",
        "arr[0] = _crosstl_ref_0;",
        "break if !(_crosstl_call_1 < 3);",
        "}",
    ]


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "int arr[2];\nint r = 0;\nif (r > 0 && bump(arr[0]) > 0) { r = 1; }",
            "in the right operand of && or ||, or a ternary branch",
        ),
        (
            "int arr[2];\nint r = 0;\nfor (int k = 0; k < 2; k = k + bump(arr[0])) { r = r + 1; }",
            "in a for-loop update: the update also runs after every continue",
        ),
        (
            "int arr[2];\nint r = g() + bump(arr[0]);",
            "the copies move the call ahead of an earlier call in the same statement",
        ),
        (
            "int r = bump(scale);",
            "cannot pass 'scale' to inout parameter a: it is a module-scope uniform",
        ),
        (
            "int r = bump(K);",
            "cannot pass 'K' to inout parameter a: it is a module-scope uniform",
        ),
        (
            "int r = bump(inBuf.data[0]);",
            "cannot write read-only GLSL buffer block resource inBuf",
        ),
        (
            "int arr[2];\nint r = bump(arr[0] + 1);",
            "cannot pass an expression to inout parameter a",
        ),
        (
            "int arr[2];\nint r = step2(arr[0], arr[1]);",
            "cannot pass 'arr' to both idx and slot of step2",
        ),
    ],
)
def test_uncopyable_arguments_fail_closed(body, message):
    with pytest.raises(ValueError, match=message):
        compute(body)


def test_writes_to_a_uniform_or_constant_are_refused():
    with pytest.raises(
        ValueError, match="cannot write 'scale': it is a module-scope uniform"
    ):
        compute("scale = 3;")


FORMS = """
Counter c;
c.count = 5;
c.pair = ivec2(1, 1);
int r0 = bump(c.count);
inc2(c.pair);
ivec3 v = ivec3(10, 20, 30);
bump(v.y);
int i = 2;
put(v[i], 9);
inc2(v.xz);
int arr[3];
arr[0] = 0; arr[1] = 1; arr[2] = 2;
int q = 0;
int r1 = step2(q, arr[q]);
tile[1] = 1;
tile[2] = 2;
int r2 = pair(poke(5), tile[1]);
int r3 = bump(arr[2]) + bump(arr[1]);
int n = 0;
while (below(arr[0], 103)) { n = n + 1; }
int r4 = resetTile(tile[2]);
outBuf.values[0] = c.count + r0 + r1 + r2 + r3 + r4 + n + v.x + q;
"""


def test_copied_argument_forms_validate_with_naga(tmp_path):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "forms.wgsl"
    path.write_text(compute(FORMS), encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr[-2000:]

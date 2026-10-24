"""WGSL lowering of written value parameters and inout/out parameters.

WGSL parameters are immutable (WGSL 11.1), so ``c += 1`` on a parameter used
to be emitted verbatim and rejected by naga. ``inout``/``out`` parameters were
emitted as plain values, which silently dropped the callee's writes.
"""

import shutil
import subprocess
import textwrap

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def compute(body, helpers="", entry_parameters=""):
    source = (
        textwrap.dedent("""
        shader Probe {
            compute {
                buffer int data[];
                struct Acc { int sum; int count; };
        %s
                [numthreads(1, 1, 1)]
                void main(%s) {
        %s
                }
            }
        }
        """)
        % (
            textwrap.indent(helpers, " " * 8),
            entry_parameters,
            textwrap.indent(body, " " * 12),
        )
    )
    return WGSLCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def function_lines(output, name):
    """The signature and stripped body lines of one emitted function."""
    start = output.index(f"fn {name}(")
    lines = []
    for line in output[start:].splitlines():
        lines.append(line.strip())
        if line == "}":
            break
    return lines


TOUCH = "void touch(inout int x, int y) { y = y * 2; x = x + y; }"


def test_written_value_parameter_is_copied_into_a_var():
    output = compute(
        "data[0] = bump(1, 2);",
        "int bump(int c, int d) { c += 1; d++; return c * 10 + d; }",
    )
    assert function_lines(output, "bump")[:3] == [
        "fn bump(c_in: i32, d_in: i32) -> i32 {",
        "var c: i32 = c_in;",
        "var d: i32 = d_in;",
    ]


def test_unwritten_value_parameter_is_unchanged():
    output = compute(
        "data[0] = twice(3);", "int twice(int c) { int d = c * 2; return d; }"
    )
    assert function_lines(output, "twice")[:2] == [
        "fn twice(c: i32) -> i32 {",
        "var d: i32 = (c * 2);",
    ]


def test_incoming_name_avoids_existing_identifiers():
    output = compute(
        "data[0] = bump(1, 2);",
        "int bump(int c, int c_in) { c = c_in + 1; return c; }",
    )
    assert function_lines(output, "bump")[:2] == [
        "fn bump(c_in_2: i32, c_in: i32) -> i32 {",
        "var c: i32 = c_in_2;",
    ]


def test_inout_parameter_becomes_a_function_pointer():
    output = compute("int a = 3;\ntouch(a, 4);\ndata[0] = a;", TOUCH)
    assert function_lines(output, "touch") == [
        "fn touch(x: ptr<function, i32>, y_in: i32) {",
        "var y: i32 = y_in;",
        "y = (y * 2);",
        "(*x) = ((*x) + y);",
        "}",
    ]
    assert "touch(&a, 4);" in output


def test_out_struct_and_array_parameters_use_pointers():
    helpers = textwrap.dedent("""
        void addTo(inout Acc acc, int v) { acc.sum += v; acc.count++; }
        void fill(out int r[2]) { r[0] = 1; r[1] = 2; }
        """)
    body = "Acc acc;\naddTo(acc, 5);\nint r[2];\nfill(r);\ndata[0] = acc.sum + r[1];"
    output = compute(body, helpers)
    assert function_lines(output, "addTo")[:3] == [
        "fn addTo(acc: ptr<function, Acc>, v: i32) {",
        "(*acc).sum += v;",
        "(*acc).count += 1;",
    ]
    assert function_lines(output, "fill")[:2] == [
        "fn fill(r: ptr<function, array<i32, 2>>) {",
        "(*r)[0] = 1;",
    ]
    assert "addTo(&acc, 5);" in output
    assert "fill(&r);" in output


def test_pointer_parameter_is_passed_through_and_value_parameter_is_copied():
    helpers = TOUCH + textwrap.dedent("""
        void twice(inout int z) { touch(z, 1); }
        int viaValue(int w) { touch(w, 5); return w; }
        """)
    output = compute("int t = 1;\ntwice(t);\ndata[0] = viaValue(t);", helpers)
    assert function_lines(output, "twice")[:2] == [
        "fn twice(z: ptr<function, i32>) {",
        "touch(z, 1);",
    ]
    assert function_lines(output, "viaValue")[:3] == [
        "fn viaValue(w_in: i32) -> i32 {",
        "var w: i32 = w_in;",
        "touch(&w, 5);",
    ]


def test_written_entry_point_parameter_is_copied():
    output = compute(
        "gid.x = gid.x + 9u;\ndata[0] = int(gid.x);",
        entry_parameters="uvec3 gid @ gl_GlobalInvocationID",
    )
    assert function_lines(output, "compute_main")[:2] == [
        "fn compute_main(@builtin(global_invocation_id) gid_in: vec3<u32>) {",
        "var gid: vec3<u32> = gid_in;",
    ]


@pytest.mark.parametrize(
    "body, helpers, message",
    [
        (
            "Acc acc;\ntouch(acc.sum, 1);",
            TOUCH,
            "cannot pass a part of 'acc' to inout parameter x of touch",
        ),
        (
            "int arr[2];\ntouch(arr[0], 1);",
            TOUCH,
            "cannot pass a part of 'arr' to inout parameter x of touch",
        ),
        ("touch(data[0], 1);", TOUCH, "a part of 'data'"),
        (
            "touch(counter, 1);",
            "shared int counter;\n" + TOUCH,
            "cannot pass 'counter' .* is a module-scope variable",
        ),
        (
            "int a = 1;\nswap2(a, a);",
            "void swap2(inout int p, inout int q) { int t = p; p = q; q = t; }",
            "cannot pass 'a' to both p and q of swap2",
        ),
        (
            "int a = 1;\nshadow(a);",
            "void shadow(inout int p) { { int p = 2; } p = 3; }",
            "a local variable in the body shadows it",
        ),
        (
            "data[0] = 1;",
            "",
            "inout parameter v of entry point main",
        ),
    ],
)
def test_unlowerable_reference_parameters_fail_closed(body, helpers, message):
    entry_parameters = "inout int v" if "entry point" in message else ""
    with pytest.raises(ValueError, match=message):
        compute(body, helpers, entry_parameters=entry_parameters)


def test_entry_point_out_parameter_from_hlsl_fails_closed():
    from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
    from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
    from crosstl.backend.DirectX.DirectxParser import HLSLParser

    hlsl = textwrap.dedent("""
        void PSMain(float4 pos : SV_Position, out float4 color : SV_Target) {
            color = pos * 0.5;
        }
        """)
    crossgl = HLSLToCrossGLConverter().generate(
        HLSLParser(HLSLLexer(hlsl).tokenize()).parse()
    )
    with pytest.raises(ValueError, match="out parameter color of entry point"):
        WGSLCodeGen().generate(Parser(Lexer(crossgl).get_tokens()).parse())


def test_lowered_program_validates_with_naga(tmp_path):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    helpers = TOUCH + textwrap.dedent("""
        int bump(int c, int d) { c += 1; d++; return c * 10 + d; }
        void addTo(inout Acc acc, int v) { acc.sum += v; acc.count++; }
        void twice(inout int z) { touch(z, 1); touch(z, 2); }
        int viaValue(int w) { touch(w, 5); return w; }
        """)
    body = textwrap.dedent("""
        int a = 3;
        touch(a, 4);
        Acc acc;
        acc.sum = 0; acc.count = 0;
        addTo(acc, 5);
        twice(a);
        data[0] = bump(1, 2) + a + acc.sum + viaValue(a);
        """)
    path = tmp_path / "probe.wgsl"
    path.write_text(compute(body, helpers))
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr

"""WGSL lowering of conditional expressions on structs, arrays and matrices.

WGSL ``select`` takes scalars and vectors only (WGSL section 17.3). A ternary
whose branches are structs, arrays or matrices was printed as ``select(...)``,
which naga rejects with "unexpected argument type for `select` call".
"""

import shutil
import subprocess
import textwrap

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def compute(body):
    source = textwrap.dedent("""
        shader Probe {
            struct Pair {
                float a;
                float b;
            };
            compute {
                buffer float data[];
                [numthreads(1, 1, 1)]
                void main() {
        %s
                }
            }
        }
        """) % textwrap.indent(body, " " * 12)
    return WGSLCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def body_lines(output):
    entry = output[output.index("fn compute_main") :]
    return [line.strip() for line in entry.splitlines()[1:] if line.strip()]


def follows(lines, *expected):
    count = len(expected)
    return any(
        tuple(lines[i : i + count]) == expected for i in range(len(lines) - count + 1)
    )


def naga_accepts(output, tmp_path):
    if shutil.which("naga") is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "probe.wgsl"
    path.write_text(output)
    result = subprocess.run(
        ["naga", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


PAIRS = "Pair p;\np.a = data[0];\np.b = 1.0;\nPair q;\nq.a = 2.0;\nq.b = 3.0;\n"


def test_struct_initializer_becomes_if_else(tmp_path):
    output = compute(PAIRS + "Pair r = data[1] > 7.0 ? p : q;\ndata[2] = r.a + r.b;")
    lines = body_lines(output)
    assert not any("select(" in line for line in lines)
    assert follows(
        lines,
        "var r: Pair;",
        "if ((data[1] > 7.0))     {",
        "r = p;",
        "} else     {",
        "r = q;",
        "}",
    )
    naga_accepts(output, tmp_path)


def test_struct_assignment_becomes_if_else(tmp_path):
    output = compute(PAIRS + "Pair r;\nr = data[1] > 7.0 ? p : q;\ndata[2] = r.b;")
    lines = body_lines(output)
    assert not any("select(" in line for line in lines)
    assert follows(lines, "r = p;", "} else     {", "r = q;")
    naga_accepts(output, tmp_path)


def test_nested_struct_ternary_lowers_each_level(tmp_path):
    output = compute(
        PAIRS + "Pair r = data[1] > 7.0 ? p : data[1] > 3.0 ? q : p;\ndata[2] = r.a;"
    )
    lines = body_lines(output)
    assert not any("select(" in line for line in lines)
    assert lines.count("r = p;") == 2
    assert lines.count("r = q;") == 1
    naga_accepts(output, tmp_path)


def test_matrix_initializer_becomes_if_else(tmp_path):
    output = compute(
        "mat2 m = mat2(1.0);\nmat2 n = mat2(2.0);\n"
        "mat2 k = data[0] > 0.0 ? m : n;\ndata[1] = k[0][0];"
    )
    lines = body_lines(output)
    assert not any("select(" in line for line in lines)
    assert follows(lines, "k = m;", "} else     {", "k = n;")
    naga_accepts(output, tmp_path)


def test_scalar_and_vector_ternaries_keep_select(tmp_path):
    output = compute(
        "float x = data[0] > 0.0 ? 1.0 : 2.0;\n"
        "vec2 v = data[0] > 0.0 ? vec2(1.0) : vec2(2.0);\n"
        "data[1] = x + v.y;"
    )
    lines = body_lines(output)
    assert "var x: f32 = select(2.0, 1.0, (data[0] > 0.0));" in lines
    assert any(line.startswith("var v: vec2<f32> = select(") for line in lines)
    naga_accepts(output, tmp_path)


def test_struct_ternary_inside_an_expression_is_refused():
    with pytest.raises(ValueError, match="WGSL select takes scalars and vectors"):
        compute(PAIRS + "data[2] = (data[1] > 7.0 ? p : q).a;")


def test_struct_initializer_that_reads_the_name_it_declares_is_refused():
    # The initializer reads the outer r; after the lowering it would read the new one.
    with pytest.raises(ValueError, match="reads a variable named r"):
        compute(
            PAIRS
            + "Pair r = p;\n{\n    Pair r = data[1] > 7.0 ? q : r;\n"
            + "    data[2] = r.a;\n}\ndata[3] = r.a;"
        )

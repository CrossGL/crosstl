"""Rust output value-initializes the elements of an enumeration array to its zero-valued variant (#4189).

``M arr[2] = {};`` value-initializes each element to 0, and ``M arr[2] = { {} };`` the elements the list leaves out.
For ``enum M { A = 2, B = 0 }`` that is ``M::B``, as ``M m = {};`` is. Rust output filled them with
``Default::default()``, whose ``impl Default for M`` returns the first variant, ``M::A``. An enumeration with no
zero-valued variant is refused with ``RustEnumEmptyBraceError``, as for ``M m = {};``.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.rust_refusals import RustEnumEmptyBraceError

SHADER = textwrap.dedent("""
    shader P {
        enum M { A = 2, B = 0 };
        int probe() {
            M arr[2] = INITIALIZER;
            return int(arr[1]);
        }
        compute {
            layout(local_size_x = 1) in;
            void main() { int v = probe(); }
        }
    }
""")


def translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="rust", format_output=False)


@pytest.mark.parametrize(
    "initializer, expected",
    [
        ("{}", "[M::B, M::B]"),
        ("{ {} }", "[M::B, M::B]"),
        ("{ {}, {} }", "[M::B, M::B]"),
        ("{M::A}", "[M::A, M::B]"),
    ],
)
def test_enum_array_elements_are_the_zero_variant(tmp_path, initializer, expected):
    output = translate(tmp_path, SHADER.replace("INITIALIZER", initializer))
    assert f"let arr: [M; 2] = {expected};" in output
    assert "Default::default()" not in output


def test_nested_enum_array_elements_are_the_zero_variant(tmp_path):
    source = SHADER.replace("M arr[2] = INITIALIZER;", "M arr[2][3] = { {M::A} };")
    output = translate(tmp_path, source)
    assert "let arr: [[M; 3]; 2] = [[M::A, M::B, M::B], [M::B; 3]];" in output
    assert "Default::default()" not in output


def test_static_enum_array_elements_are_the_zero_variant(tmp_path):
    source = SHADER.replace(
        "    int probe() {\n        M arr[2] = INITIALIZER;",
        "    M table[2] = {};\n    int probe() {\n        M arr[2] = table;",
    )
    output = translate(tmp_path, source)
    assert "static TABLE: [M; 2] = [M::B, M::B];" in output
    assert "Default::default()" not in output


@pytest.mark.parametrize("initializer", ["{}", "{ {} }", "{M::A}"])
def test_enum_array_without_a_zero_variant_is_refused(tmp_path, initializer):
    source = SHADER.replace("A = 2, B = 0", "A = 2, B = 5").replace(
        "INITIALIZER", initializer
    )
    with pytest.raises(RustEnumEmptyBraceError) as info:
        translate(tmp_path, source)
    assert info.value.enum_name == "M"
    assert "no variant has the value 0" in str(info.value)


def test_array_of_an_enum_with_data_is_refused(tmp_path):
    source = (
        SHADER.replace("A = 2, B = 0", "A(int), B")
        .replace("return int(arr[1]);", "return 1;")
        .replace("INITIALIZER", "{}")
    )
    with pytest.raises(RustEnumEmptyBraceError, match="variants carry data"):
        translate(tmp_path, source)


RUN = textwrap.dedent("""
    shader P {
        enum M { A = 2, B = 0 };
        M table[2] = {};
        int probe() {
            M arr[2] = {};
            M short_list[2] = { {} };
            M grid[2][2] = { {M::A} };
            return int(arr[0]) + int(arr[1]) + int(short_list[1]) * 10
                + int(grid[0][1]) * 100 + int(grid[1][0]) * 1000 + int(table[1]) * 10000;
        }
        compute {
            layout(local_size_x = 1) in;
            void main() { int v = probe(); }
        }
    }
""")


def test_enum_array_empty_brace_initializers_compile_to_the_zero_variant(tmp_path):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    code = translate(tmp_path, RUN)
    program = tmp_path / "probe.rs"
    program.write_text(
        code[code.index("#[") : code.index("// Compute Shader")]
        + '\nfn main() { println!("{}", probe()); }\n',
        encoding="utf-8",
    )
    binary = tmp_path / "probe"
    compiled = subprocess.run(
        [rustc, "--edition", "2021", "-A", "warnings", str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert int(ran.stdout) == 0

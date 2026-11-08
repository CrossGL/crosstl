"""Rust output writes an empty brace initializer of a plain enumeration as its zero-valued variant (#3820).

``Mode m = {};`` is the value 0. Rust has no integer-valued enumeration, so the variant whose discriminant is 0
stands for it: ``let m: Mode = Mode::First;`` for ``enum Mode { First, Second }``. When no variant is 0, as in
``enum Mode { First = 3, Second = 5 }``, translation raises ``RustEnumEmptyBraceError``.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.rust_refusals import RustEnumEmptyBraceError

SHADER = textwrap.dedent("""
    shader P {
        enum Mode { First, Second }
        compute {
            buffer float* outp;
            void main() {
                Mode m = {};
                outp[0] = float(int(m));
            }
        }
    }
""")

EXPLICIT_ZERO = textwrap.dedent("""
    shader P {
        enum Mode { First = 1, Second = 0 }
        compute {
            buffer float* outp;
            void main() {
                Mode m = {};
                outp[0] = float(int(m));
            }
        }
    }
""")

NO_ZERO = textwrap.dedent("""
    shader P {
        enum Mode { First = 3, Second = 5 }
        compute {
            buffer float* outp;
            void main() {
                Mode m = {};
                outp[0] = float(int(m));
            }
        }
    }
""")


def translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="rust", format_output=False)


def test_rust_empty_brace_enum_initializer_is_the_zero_variant(tmp_path):
    output = translate(tmp_path, SHADER)
    assert "let m: Mode = Mode::First;" in output
    assert "let m: Mode = [];" not in output


def test_rust_empty_brace_enum_initializer_picks_the_variant_whose_value_is_zero(
    tmp_path,
):
    output = translate(tmp_path, EXPLICIT_ZERO)
    assert "let m: Mode = Mode::Second;" in output
    assert "let m: Mode = Mode::First;" not in output


def test_rust_empty_brace_enum_initializer_without_a_zero_variant_is_refused(tmp_path):
    with pytest.raises(RustEnumEmptyBraceError) as info:
        translate(tmp_path, NO_ZERO)
    assert info.value.enum_name == "Mode"
    assert info.value.project_diagnostic_code == (
        "project.translate.rust-enum-empty-brace-unsupported"
    )


def test_rust_empty_brace_enum_initializer_counts_on_from_a_negative_value(tmp_path):
    output = translate(tmp_path, NO_ZERO.replace("First = 3, Second = 5", "A = -1, B"))
    assert "let m: Mode = Mode::B;" in output


def test_rust_empty_brace_initializer_of_an_enum_with_data_is_refused(tmp_path):
    source = NO_ZERO.replace("First = 3, Second = 5", "Value(int), Missing").replace(
        "outp[0] = float(int(m));", "outp[0] = 1.0;"
    )
    with pytest.raises(RustEnumEmptyBraceError, match="variants carry data"):
        translate(tmp_path, source)


RUN = textwrap.dedent("""
    shader P {
        enum Mode { First = 2, Second = 0 }
        struct Holder { Mode m; float x; };
        Mode pick() { return {}; }
        int take(Mode m) { return int(m); }
        int probe() {
            Mode a = {};
            Mode arr[2] = { {}, {} };
            Holder h;
            h.m = Mode::First;
            h.m = {};
            return int(a) + int(arr[1]) * 10 + int(pick()) * 100 + take({}) * 1000
                + int(h.m) * 10000;
        }
        compute {
            layout(local_size_x = 1) in;
            void main() { int v = probe(); }
        }
    }
""")


def test_rust_empty_brace_enum_initializers_compile_to_the_zero_variant(tmp_path):
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

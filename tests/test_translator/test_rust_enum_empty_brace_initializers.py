"""Rust output writes an empty brace initializer of a plain enumeration as its zero-valued variant (#3820).

``Mode m = {};`` is the value 0. Rust has no integer-valued enumeration, so the variant whose discriminant is 0
stands for it: ``let m: Mode = Mode::First;`` for ``enum Mode { First, Second }``. When no variant is 0, as in
``enum Mode { First = 3, Second = 5 }``, translation raises ``RustEnumEmptyBraceError``.
"""

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

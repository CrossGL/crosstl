"""Tests that ``var x = <expr>`` translates correctly for the WGSL backend.

Covers the cases described in the issue: a local variable whose initialiser is a
``const`` identifier, a binary expression involving a ``const``, or a call to a
component-wise numeric built-in such as ``sqrt`` or ``abs``.
"""

import textwrap

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def parse_shader(source):
    return Parser(Lexer(source).get_tokens()).parse()


CONST_TIMES_LITERAL_SHADER = textwrap.dedent("""
    shader main {
        const float BASE = 2.0;
        compute {
            buffer float data[];
            void main() {
                var x = BASE * 3.0;
                data[0] = x;
            }
        }
    }
""")

SQRT_SHADER = textwrap.dedent("""
    shader main {
        compute {
            buffer float data[];
            void main() {
                var x = sqrt(4.0);
                data[0] = x;
            }
        }
    }
""")

ABS_SHADER = textwrap.dedent("""
    shader main {
        compute {
            buffer float data[];
            void main() {
                var x = abs(-1.0);
                data[0] = x;
            }
        }
    }
""")

LITERAL_FLOAT_SHADER = textwrap.dedent("""
    shader main {
        compute {
            buffer float data[];
            void main() {
                var x = 2.0;
                data[0] = x;
            }
        }
    }
""")


def test_wgsl_untyped_var_from_const_times_literal():
    """``var x = BASE * 3.0`` translates to ``var x: f32 = ...`` when BASE is a float const."""
    generated = WGSLCodeGen().generate(parse_shader(CONST_TIMES_LITERAL_SHADER))

    assert "var x: f32 =" in generated
    assert ": void" not in generated


def test_wgsl_untyped_var_from_sqrt_call():
    """``var x = sqrt(4.0)`` translates to ``var x: f32 = ...``."""
    generated = WGSLCodeGen().generate(parse_shader(SQRT_SHADER))

    assert "var x: f32 =" in generated
    assert ": void" not in generated


def test_wgsl_untyped_var_from_abs_call():
    """``var x = abs(-1.0)`` translates to ``var x: f32 = ...``."""
    generated = WGSLCodeGen().generate(parse_shader(ABS_SHADER))

    assert "var x: f32 =" in generated
    assert ": void" not in generated


def test_wgsl_untyped_var_from_literal_float():
    """``var x = 2.0`` translates to ``var x: f32 = 2.0`` (pre-existing behaviour, not broken)."""
    generated = WGSLCodeGen().generate(parse_shader(LITERAL_FLOAT_SHADER))

    assert "var x: f32 = 2.0;" in generated
    assert ": void" not in generated


def test_wgsl_no_error_on_const_times_literal():
    """``var x = BASE * 3.0`` does not raise ValueError."""
    # This was the main failure mode: a ValueError about void type.
    generated = WGSLCodeGen().generate(parse_shader(CONST_TIMES_LITERAL_SHADER))
    assert "const BASE: f32 = 2.0;" in generated


def test_wgsl_no_error_on_sqrt():
    """``var x = sqrt(4.0)`` does not raise ValueError."""
    generated = WGSLCodeGen().generate(parse_shader(SQRT_SHADER))
    assert "sqrt" in generated

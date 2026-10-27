"""A signed WGSL shift amount is converted to ``u32`` whatever its expression form.

WGSL takes a shift amount of type ``u32`` (or an abstract-integer literal); the
generator converts every signed spelling of the amount, and naga rejects the
output when one slips through (WGSL, "Bit Expressions").
"""

import shutil
import subprocess

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate(tmp_path, body):
    source = f"""
    shader Probe {{
        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer int* iin;
            buffer uint* uout;
            void main() {{
                int n = iin[0];
                uint m = uout[0];
                {body}
            }}
        }}
    }}
    """
    generated = WGSLCodeGen().generate(Parser(Lexer(source).tokens).parse())
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(generated)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return generated


def test_bitwise_shift_amount_becomes_u32(tmp_path):
    generated = generate(
        tmp_path, "uout[0] = uint((1 << (n & 3)) + (n >> (n | 4)) + (2 >> (n ^ 1)));"
    )

    assert "u32((n & 3))" in generated
    assert "u32((n | 4))" in generated
    assert "u32((n ^ 1))" in generated


def test_tilde_shift_amount_becomes_u32(tmp_path):
    generated = generate(tmp_path, "uout[0] = uint(1 << ~n);")

    assert "u32(~n)" in generated


def test_ternary_shift_amount_becomes_u32(tmp_path):
    generated = generate(tmp_path, "uout[0] = uint(4 << (n > 0 ? 2 : n));")

    assert "u32(select(n, 2, (n > 0)))" in generated


def test_unsigned_bitwise_shift_amount_stays_unwrapped(tmp_path):
    generated = generate(tmp_path, "uout[0] = (m << (m & 3u)) + (1u >> (m | 4u));")

    assert "(m & 3u)" in generated
    assert "(m | 4u)" in generated
    assert "u32((m & 3u))" not in generated


def test_vector_bitwise_shift_amount_becomes_u32_vector(tmp_path):
    generated = generate(
        tmp_path,
        "int4 v = int4(1, 2, 3, 4);\n"
        "                int4 w = int4(5, 6, 7, 8);\n"
        "                int4 r = v << (v & w);",
    )

    assert "vec4<u32>((v & w))" in generated


def test_bitwise_scalar_operand_of_vector_is_replicated(tmp_path):
    generated = generate(tmp_path, "int4 v = int4(1, 2, 3, 4); int4 r = v & (n | 3);")

    assert "vec4<i32>((n | 3))" in generated

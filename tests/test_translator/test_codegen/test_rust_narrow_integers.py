"""Rust output computes 8- and 16-bit integer arithmetic after integer promotion, as the IR defines it (X3).

The Rust generator crashed with an untyped ``KeyError: 'u8'`` when it ranked ``u8`` or ``i8`` operands, wrote the
IR spellings ``int16``/``uint8`` as Rust types, and computed ``i16 + i16`` in ``i16``. The IR promotes an operand
narrower than 32 bits to ``int`` (``semantics.hub_types.promote``), so ``30000 + 30000`` on two ``int16`` values is
60000 and ``3 - 5`` on two ``uint16`` values is -2.
"""

import subprocess

import pytest

import crosstl
from tests.test_translator.test_codegen.test_rust_codegen import (
    generate_code,
    parse_code,
    rust_smoke_source,
    rustc_or_skip,
    tokenize_code,
)

HELPERS = """
shader NarrowArithmetic {
    int add16(int16 a, int16 b) { return a + b; }
    int sub16u(uint16 a, uint16 b) { return a - b; }
    int mul8(int8 a, uint8 b) { return a * b; }
    uint8 wrap8(uint8 a, uint8 b) { return uint8(a + b); }
    int shl8(uint8 a) { return a << 4; }
}
"""


def rust(code):
    return generate_code(parse_code(tokenize_code(code)))


def test_ir_narrow_integer_spellings_are_rust_types():
    generated = rust(HELPERS)

    for foreign in ("int16", "uint16", "int8", "uint8"):
        assert f": {foreign}" not in generated
        assert f"{foreign}(" not in generated
    assert "pub fn add16(a: i16, b: i16) -> i32" in generated
    assert "pub fn wrap8(a: u8, b: u8) -> u8" in generated
    assert "((a as i32) + (b as i32))" in generated


def test_narrow_integer_arithmetic_runs_with_integer_promotion(tmp_path):
    source = rust_smoke_source(rust(HELPERS)) + (
        "\nfn main() {\n"
        '    println!("{}", add16(30000, 30000));\n'
        '    println!("{}", sub16u(3, 5));\n'
        '    println!("{}", mul8(-100, 200));\n'
        '    println!("{}", wrap8(200, 100));\n'
        '    println!("{}", shl8(255));\n'
        "}\n"
    )
    source_path = tmp_path / "narrow.rs"
    binary = tmp_path / "narrow"
    source_path.write_text(source, encoding="utf-8")
    build = subprocess.run(
        [rustc_or_skip(), "--edition=2021", "-o", str(binary), str(source_path)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)

    assert run.stdout.split() == ["60000", "-2", "-20000", "44", "4080"]


@pytest.mark.parametrize(
    "source, name",
    [
        (
            "__global__ void k(unsigned char* out, const char* in_, int n) {\n"
            "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (i < n) {\n"
            "        uchar4 v = make_uchar4(1, 2, 3, 4);\n"
            "        out[i] = v.x + v.y + in_[i];\n"
            "    }\n"
            "}\n",
            "bytes.cu",
        ),
    ],
)
def test_imported_byte_arithmetic_translates_to_rust(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")

    generated = crosstl.translate(str(path), backend="rust", format_output=False)

    assert "(((v.x as i32) + (v.y as i32))" in generated
    assert "as u8)" in generated

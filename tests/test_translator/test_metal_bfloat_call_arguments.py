"""Metal bfloat arguments of user functions convert only computed values.

Metal has no implicit conversion from float to bfloat, so a computed argument
of a bfloat parameter is constructed explicitly. A value that already has the
bfloat type, such as a user function result or a dereferenced bfloat buffer,
is passed unchanged.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader BfloatCallArguments {
    bfloat16 identity(bfloat16 value) {
        return value;
    }

    bfloat16 combine(bfloat16 a, bfloat16 b) {
        return a * b;
    }

    bool positive(bfloat16 value) {
        return value > bfloat16(0.0);
    }

    compute {
        @ stage_entry
        void reduce_row(StructuredBuffer<bfloat16> row @buffer(0), RWStructuredBuffer<float> values @buffer(1)) {
            float f = values[0];
            bfloat16 b = bfloat16(values[1]);
            bfloat16 c = bfloat16(values[2]);
            bfloat16 r0 = combine(identity(b), c);
            bfloat16 r1 = combine(f, c);
            bool r2 = positive((*row));
            buffer_store(values, 3, float(r0) + float(r1) + float(r2));
        }
    }
}
"""


def _metal():
    return MetalCodeGen().generate(Parser(Lexer(PROGRAM).get_tokens()).parse())


def _initializers(code):
    lines = {}
    for line in code.splitlines():
        match = re.match(r"\s*\w+ (r\d) = (.+);$", line)
        if match:
            lines[match.group(1)] = match.group(2)
    return lines


def test_user_function_result_is_passed_without_conversion():
    assert "combine(identity(b), c)" in _initializers(_metal())["r0"]


def test_dereferenced_bfloat_buffer_is_passed_without_conversion():
    assert _initializers(_metal())["r2"] == "positive(*row)"


def test_float_argument_is_constructed_as_bfloat():
    assert "combine(bfloat(f), c)" in _initializers(_metal())["r1"]


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="requires the Metal compiler")
def test_bfloat_arguments_compile(tmp_path):
    source = tmp_path / "arguments.metal"
    source.write_text(_metal(), encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.1",
            "-Werror",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "arguments.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

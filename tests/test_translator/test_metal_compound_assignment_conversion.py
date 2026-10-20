"""Metal compound assignments convert the right operand like the target's family.

Floating targets, bfloat included, take the right operand converted to the
target type, as plain assignments do. Integer targets combined with a floating
operand keep the operand unconverted so the operation runs in the common type.
"""

import re

import pytest

from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader CompoundConversion {
    compute {
        layout(std430, binding = 0) buffer float values[];
        [numthreads(1, 1, 1)]
        void main() {
            float f = values[0];
            bfloat16 b = bfloat16(values[1]);
            bfloat16 c = bfloat16(values[2]);
            int i = int(values[3]);
            b += f;
            c -= b;
            i *= f;
            values[4] = float(b) + float(c) + float(i);
        }
    }
}
"""


def _updates():
    code = MetalCodeGen().generate(Parser(Lexer(PROGRAM).get_tokens()).parse())
    updates = {}
    for line in code.splitlines():
        match = re.match(r"\s*(\w+) ([-+*]=) (.+);$", line)
        if match:
            updates[match.group(1)] = (match.group(2), match.group(3))
    return updates


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        # A float operand is rounded to the bfloat target before combining.
        ("b", ("+=", "bfloat(f)")),
        # A bfloat operand of a bfloat target keeps the same conversion.
        ("c", ("-=", "bfloat(b)")),
    ],
)
def test_bfloat_targets_convert_the_operand_like_other_floating_targets(
    target, expected
):
    assert _updates()[target] == expected


def test_integer_target_keeps_a_floating_operand_in_the_common_type():
    # Converting 'f' to int first would truncate it before the multiply.
    assert _updates()["i"] == ("*=", "f")

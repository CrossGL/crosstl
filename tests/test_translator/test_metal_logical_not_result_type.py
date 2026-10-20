"""Metal result type of logical negation at implicit-conversion boundaries."""

import re

from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader LogicalNot {
    compute {
        layout(std430, binding = 0) buffer int values[];
        [numthreads(1, 1, 1)]
        void main() {
            vec3 v = vec3(float(values[0]), 0.0, 2.0);
            ivec3 iv = ivec3(values[1], 0, values[2]);
            bvec3 nv = !v;
            bvec2 nt = !v;
            bvec3 ni = !iv;
            bool ns = !v.y;
            values[3] = int(nv.y) + int(nt.x) + int(ni.z) + int(ns);
        }
    }
}
"""

# A bool conversion wrapped around a negation, ``bool3 nv = bool3(!...)``.
CONVERTED_NEGATION = re.compile(r"^bool\d? \w+ = bool\d?\(\(?!")


def _initializers():
    code = MetalCodeGen().generate(Parser(Lexer(PROGRAM).get_tokens()).parse())
    lines = {}
    for line in code.splitlines():
        match = re.match(r"\s*(bool\d?) (n[vtis]) = ", line)
        if match:
            lines[match.group(2)] = (match.group(1), line.strip())
    return lines


def test_logical_not_takes_the_bool_shape_of_its_operand():
    lines = _initializers()

    assert lines["nv"][0] == "bool3"
    assert lines["ni"][0] == "bool3"
    assert lines["ns"][0] == "bool"
    for name in ("nv", "ni", "ns"):
        # The negation is already bool, so no component conversion is added.
        assert not CONVERTED_NEGATION.match(lines[name][1]), lines[name][1]


def test_logical_not_truncation_spells_only_the_swizzle():
    declared, line = _initializers()["nt"]

    assert declared == "bool2"
    assert line.endswith(").xy;"), line
    assert not CONVERTED_NEGATION.match(line), line

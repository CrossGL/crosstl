"""Bit casts and the rounding, remainder and fused built-ins fold exactly.

``const float INF = uintBitsToFloat(0x7f800000u)`` is how CrossTL's own GLSL
output spells infinity, and HLSL spells it ``asfloat(0x7f800000)``. The SPIR-V
generator refused such module-scope constants, along with ``round``,
``roundEven``, ``mod``, ``fmod``, ``fma`` and ``ldexp`` initializers.

The folded values follow the source languages:

* bit casts reinterpret the 32-bit pattern;
* ``round`` refuses a fraction of exactly 0.5, whose direction GLSL leaves to
  the implementation;
* ``mod`` is ``x - y * floor(x / y)``;
* ``fma`` rounds once;
* ``ldexp`` refuses results that overflow or would be subnormal.

Zero results keep the operand's sign.
"""

import math
import re
import shutil
import subprocess
import textwrap
from fractions import Fraction

import pytest

import crosstl
from crosstl.translator.codegen.scalar_constant_folding import (
    ScalarConstant,
    fold_math_builtin,
    fused_multiply_add,
    round_fraction,
    round_to_binary32,
)


def _f(value):
    return ScalarConstant("float", value)


def _u(value):
    return ScalarConstant("uint", value)


def _i(value):
    return ScalarConstant("int", value)


def _same(folded, expected):
    # ScalarConstant equality treats -0.0 and 0.0 as equal; compare signs too.
    assert folded == expected
    if expected.kind == "float":
        assert math.copysign(1.0, folded.value) == math.copysign(1.0, expected.value)


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("uintBitsToFloat", [_u(0x7F800000)], _f(math.inf)),
        ("uintBitsToFloat", [_u(0x3F800000)], _f(1.0)),
        ("intBitsToFloat", [_i(-0x80000000)], _f(-0.0)),
        ("asfloat", [_i(0x7F7FFFFF)], _f(3.4028234663852886e38)),
        ("floatBitsToUint", [_f(-0.0)], _u(0x80000000)),
        ("floatBitsToInt", [_f(-1.0)], _i(-0x40800000)),
        ("asuint", [_f(1.0)], _u(0x3F800000)),
        ("asint", [_u(0xFFFFFFFF)], _i(-1)),
        ("round", [_f(2.4)], _f(2.0)),
        ("round", [_f(-0.4)], _f(-0.0)),
        ("roundEven", [_f(2.5)], _f(2.0)),
        ("roundEven", [_f(3.5)], _f(4.0)),
        ("ceil", [_f(-0.5)], _f(-0.0)),
        ("trunc", [_f(-0.7)], _f(-0.0)),
        ("floor", [_f(-0.0)], _f(-0.0)),
        ("mod", [_f(-1.0), _f(3.0)], _f(2.0)),
        ("fmod", [_f(-1.0), _f(3.0)], _f(-1.0)),
        ("ldexp", [_f(1.5), _i(3)], _f(12.0)),
        ("ldexp", [_f(1.0), _i(-126)], _f(2.0**-126)),
    ],
)
def test_builtins_fold_exactly(name, arguments, expected):
    _same(fold_math_builtin(name, arguments), expected)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        # A NaN payload other than the canonical quiet NaN has no literal.
        ("uintBitsToFloat", [_u(0xFFFFFFFF)]),
        ("floatBitsToUint", [_f(math.nan)]),
        ("intBitsToFloat", [_f(1.5)]),
        ("asuint", [ScalarConstant("double", 1.0)]),
        ("round", [_f(2.5)]),
        ("round", [_f(-0.5)]),
        ("mod", [_f(1.0), _f(0.0)]),
        ("fmod", [_f(1.0), _f(0.0)]),
        ("ldexp", [_f(1.0), _i(128)]),
        ("ldexp", [_f(1.0), _i(-127)]),
        ("ldexp", [_f(1.0), _f(2.0)]),
    ],
)
def test_builtins_without_a_defined_value_are_not_folded(name, arguments):
    assert fold_math_builtin(name, arguments) is None


def test_fused_multiply_add_rounds_once():
    a = round_to_binary32(1.0 + 2.0**-12)
    # a * a = 1 + 2^-11 + 2^-24 exactly; a separate multiply loses the 2^-24.
    assert fused_multiply_add("float", a, a, -1.0) == 2.0**-11 + 2.0**-24
    assert round_to_binary32(round_to_binary32(a * a) - 1.0) == 2.0**-11
    assert math.copysign(1.0, fused_multiply_add("float", -0.0, 1.0, -0.0)) < 0
    assert math.copysign(1.0, fused_multiply_add("float", -0.0, 1.0, 0.0)) > 0


def test_round_fraction_rounds_directly_to_binary32():
    # Just above the midpoint between 1 and the next binary32 value. Rounding
    # through binary64 first lands on the midpoint and then rounds down.
    value = Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**80)
    assert round_fraction(value, "float") == 1.0 + 2.0**-23
    assert round_to_binary32(float(value)) == 1.0
    assert round_fraction(Fraction(2) ** 128, "float") == math.inf
    assert round_fraction(Fraction(1, 2**149), "float") == 2.0**-149


PROGRAM = textwrap.dedent("""
    shader S {
        const float INF = uintBitsToFloat(0x7f800000u);
        const uint ONE_BITS = floatBitsToUint(1.0);
        const float R = roundEven(2.5) + mod(-1.0, 3.0) + ldexp(1.5, 3);
        const vec2 V = uintBitsToFloat(uvec2(0x3f800000u, 0x40000000u));
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                data[0] = INF + float(ONE_BITS) + R + V.y;
            }
        }
    }
    """)


def _raw_spirv(tmp_path, program):
    source = tmp_path / "constants.cgl"
    source.write_text(program, encoding="utf-8")
    # Unformatted output keeps numeric ids whether or not SPIRV-Tools is installed.
    return crosstl.translate(str(source), backend="vulkan", format_output=False)


def _operands(raw, name_or_id):
    named = re.search(rf'^OpName %(\w+) "{name_or_id}"$', raw, re.M)
    identifier = named.group(1) if named else name_or_id
    definition = re.search(rf"^%{identifier} = (Op\w+) %\w+(.*)$", raw, re.M)
    assert definition, name_or_id
    return definition.group(1), definition.group(2).split()


def test_spirv_constants_hold_the_folded_builtin_values(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    assert _operands(raw, "INF") == ("OpConstant", ["0x1p+128"])
    assert _operands(raw, "ONE_BITS") == ("OpConstant", ["1065353216"])
    assert _operands(raw, "R") == ("OpConstant", ["16.0"])
    opcode, components = _operands(raw, "V")
    assert opcode == "OpConstantComposite"
    assert [_operands(raw, c.lstrip("%"))[1] for c in components] == [
        ["1.0"],
        ["2.0"],
    ]
    module_scope = raw.split(" = OpFunction ", 1)[0]
    assert not re.search(r"= OpBitcast |= OpExtInst |= OpFMod ", module_scope)


def test_spirv_builtin_output_assembles_and_validates(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    (tmp_path / "constants.spvasm").write_text(_raw_spirv(tmp_path, PROGRAM))
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", "constants.spvasm", "-o", "c.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "c.spv"],
    ):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, cwd=tmp_path
        )
        assert result.returncode == 0, result.stderr[-2000:]

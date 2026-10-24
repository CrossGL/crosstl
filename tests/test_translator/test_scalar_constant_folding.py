"""Module-scope scalar constants fold with the target's arithmetic.

SPIR-V module scope only holds constant instructions. The SPIR-V generator used
to evaluate float initializers in binary64, folded nothing but ``+ - * /`` over
literals, and fell back to emitting arithmetic at module scope (invalid SPIR-V)
with a zero-valued constant. ``-INFINITY`` became ``0``, ``vec2(-A, A * 2.0)``
became ``vec2(0)``, ``A > 0.05`` became ``false`` and ``0xFFFFFFFFu + 2u``
became ``0``.

Folding now rounds binary32 values after every operation, wraps integers, and
refuses initializers whose value the source leaves undefined.
"""

import math
import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.scalar_constant_folding import (
    ScalarConstant,
    ScalarConstantFolder,
    convert_scalar_constant,
    fold_math_builtin,
    round_to_binary32,
    shortest_binary32_text,
)

_KINDS = {"float", "double", "int", "uint", "bool", "i64", "u64"}


def _kind(type_value):
    name = (
        type_value if isinstance(type_value, str) else getattr(type_value, "name", None)
    )
    return name if name in _KINDS else None


def _fold(expression, names=None):
    names = names or {}
    program = parse(
        "shader S { const float probe = %s; compute { void main() {} } }" % expression
    )
    folder = ScalarConstantFolder(names.get, _kind, fold_builtin=fold_math_builtin)
    return folder.fold(program.constants[0].value)


@pytest.mark.parametrize(
    ("expression", "kind", "value"),
    [
        # 2^24 + 1 rounds back to 2^24 in binary32, so the difference is 0;
        # binary64 evaluation gives 1.
        ("16777216.0 + 1.0 - 16777216.0", "float", 0.0),
        ("0.1 * 3.0", "float", round_to_binary32(0.30000001192092896)),
        ("1.0 / 0.0", "float", math.inf),
        ("-1.0 / 0.0", "float", -math.inf),
        ("0xFFFFFFFFu + 2u", "uint", 1),
        ("-2147483647 - 1", "int", -(1 << 31)),
        ("2147483647 + 1", "int", -(1 << 31)),
        ("-7 / 2", "int", -3),
        ("-7 % 2", "int", -1),
        ("~5 ^ 3", "int", -7),
        ("1u << 31", "uint", 1 << 31),
        ("-8 >> 1", "int", -4),
        ("uint(-1)", "uint", (1 << 32) - 1),
        ("int(-3.9)", "int", -3),
        ("float(7) / 2.0", "float", 3.5),
        ("0.1 > 0.05", "bool", True),
        ("1 < 2 && !(3 >= 4)", "bool", True),
        ("false ? 1.0 : 2.0", "float", 2.0),
        ("min(abs(-2.0), sqrt(2.0))", "float", round_to_binary32(math.sqrt(2.0))),
        ("clamp(7, 0, 5)", "int", 5),
        ("sin(0.5)", "float", round_to_binary32(math.sin(round_to_binary32(0.5)))),
        ("mix(0.0, 10.0, 0.25)", "float", 2.5),
    ],
)
def test_expressions_fold_with_target_arithmetic(expression, kind, value):
    folded = _fold(expression)
    assert folded == ScalarConstant(kind, value)


@pytest.mark.parametrize(
    "expression",
    [
        "1 / 0",
        "5 % 0",
        "1 << 32",
        "1 << -1",
        "(-2147483647 - 1) / -1",
        "int(1.0e10)",
        "uint(-1.0)",
        "sqrt(-1.0)",
        "log(0.0)",
        "pow(-2.0, 0.5)",
        "clamp(1.0, 2.0, 0.0)",
        "unknownName + 1.0",
    ],
)
def test_undefined_results_are_not_folded(expression):
    assert _fold(expression) is None


def test_names_resolve_through_the_resolver_and_round_to_their_kind():
    names = {"A": ScalarConstant("float", 0.1), "N": ScalarConstant("uint", 3)}
    assert _fold("A * 3.0", names) == _fold("0.1 * 3.0")
    assert _fold("N - 4u", names) == ScalarConstant("uint", (1 << 32) - 1)


def test_wide_integers_round_once_to_binary32():
    # 2^60 + 2^36 + 1 is just above the midpoint between two binary32 values;
    # rounding through binary64 first would lose the +1 and round down.
    value = (1 << 60) + (1 << 36) + 1
    folded = convert_scalar_constant(ScalarConstant("i64", value), "float")
    assert folded == ScalarConstant("float", float((1 << 60) + (1 << 37)))


def test_shortest_binary32_text_reads_back_exactly():
    for value in (0.1, 1.0 / 3.0, 3.14159265359, 1.0e-38, 3.4e38):
        rounded = round_to_binary32(value)
        assert round_to_binary32(float(shortest_binary32_text(rounded))) == rounded
    assert shortest_binary32_text(round_to_binary32(0.1)) == "0.1"
    assert round_to_binary32(1.0e39) == math.inf


PROGRAM = textwrap.dedent("""
    shader S {
        const float A = 0.1;
        const float B = 16777216.0 + 1.0 - 16777216.0;
        const float D = -INFINITY;
        const vec2 V = vec2(-A, A * 2.0);
        const uint U = 0xFFFFFFFFu + 2u;
        const bool Q = A > 0.05;
        const float Z = 0.0;
        const float NZ = -0.0;
        const float S = sqrt(2.0);
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                data[0] = B + D + V.x + V.y + float(U) + (Q ? 1.0 : 0.0) + Z + NZ + S;
            }
        }
    }
    """)


def _raw_spirv(tmp_path, program):
    source = tmp_path / "constants.cgl"
    source.write_text(program, encoding="utf-8")
    # Unformatted output keeps numeric ids whether or not SPIRV-Tools is installed.
    return crosstl.translate(str(source), backend="vulkan", format_output=False)


def _definition(raw, name):
    named = re.search(rf'^OpName %(\w+) "{name}"$', raw, re.M)
    assert named, name
    definition = re.search(rf"^%{named.group(1)} = (Op\w+) %\w+(.*)$", raw, re.M)
    assert definition, name
    return definition.group(1), definition.group(2).split()


def test_spirv_constants_hold_the_folded_values(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    assert _definition(raw, "B") == ("OpConstant", ["0.0"])
    assert _definition(raw, "D") == ("OpConstant", ["-0x1p+128"])
    assert _definition(raw, "U") == ("OpConstant", ["1"])
    assert _definition(raw, "Q") == ("OpConstantTrue", [])
    assert _definition(raw, "NZ") == ("OpConstant", ["-0.0"])
    assert _definition(raw, "S") == ("OpConstant", ["1.4142135"])

    opcode, components = _definition(raw, "V")
    assert opcode == "OpConstantComposite"
    values = []
    for component in components:
        match = re.search(rf"^{component} = OpConstant %\w+ (\S+)$", raw, re.M)
        values.append(match.group(1))
    assert values == ["-0.1", "0.2"]


def test_spirv_module_scope_holds_no_instructions(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    module_scope = raw.split(" = OpFunction ", 1)[0]
    assert not re.search(
        r"= Op(F|I|S|U)(Add|Sub|Mul|Div|Negate)|= OpExtInst |= OpFOrd", module_scope
    )


def test_undefined_initializers_are_refused(tmp_path):
    program = "shader S { const int D = 1 / 0; compute { void main() {} } }"
    with pytest.raises(ValueError, match="D has an initializer that is not a constant"):
        _raw_spirv(tmp_path, program)


def test_spirv_output_assembles_and_validates(tmp_path):
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

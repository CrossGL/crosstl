"""Element and member access of module-scope array and struct constants fold.

For ``const float X = A[1]`` over ``const float A[3] = {...}``, or
``const vec2 Z = S0.b * 2.0`` over a struct constant, the SPIR-V generator
emitted the access at module scope (invalid SPIR-V) and gave the constant the
value zero. A module-scope ``float[3](...)`` array constructor was refused as
an indirect call.

Array elements and struct members now fold to the constants they hold, and
array constructors become ``OpConstantComposite``.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.aggregate_constant_folding import (
    AggregateConstant,
    AggregateConstantFolder,
)
from crosstl.translator.codegen.scalar_constant_folding import (
    ScalarConstant,
    fold_math_builtin,
)
from crosstl.translator.codegen.vector_constant_folding import VectorConstant


def _name(type_value):
    return (
        type_value if isinstance(type_value, str) else getattr(type_value, "name", None)
    )


def _kind(type_value):
    name = _name(type_value)
    return name if name in {"float", "int", "uint", "bool"} else None


def _vector_shape(type_value):
    match = re.fullmatch(r"vec([234])", _name(type_value) or "")
    return ("float", int(match.group(1))) if match else None


def _float(value):
    return ScalarConstant("float", float(value))


def _vector(*values):
    return VectorConstant("float", tuple(_float(value) for value in values))


NAMES = {
    "A": AggregateConstant((_float(1), _float(2), _float(3))),
    "VA": AggregateConstant((_vector(1, 2), _vector(3, 4))),
    "S": AggregateConstant((_float(1), _vector(2, 3)), ("a", "b")),
    "NS": AggregateConstant(
        (AggregateConstant((_float(5), _vector(6, 7)), ("a", "b")),), ("inner",)
    ),
}


def _fold(expression):
    program = parse(
        "shader S { const float probe = %s; compute { void main() {} } }" % expression
    )
    folder = AggregateConstantFolder(
        NAMES.get,
        _kind,
        _vector_shape,
        lambda _type: None,
        fold_builtin=fold_math_builtin,
    )
    return folder.fold(program.constants[0].value)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("A[1]", _float(2)),
        ("A[2] * 2.0", _float(6)),
        ("VA[1]", _vector(3, 4)),
        ("VA[1].y", _float(4)),
        ("VA[0][1]", _float(2)),
        ("S.a", _float(1)),
        ("S.b", _vector(2, 3)),
        ("S.b.yx * 2.0", _vector(6, 4)),
        ("NS.inner.b.x", _float(6)),
        ("max(A[0], S.b.y)", _float(3)),
    ],
)
def test_aggregate_elements_and_members_fold(expression, expected):
    assert _fold(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "A[3]",
        "A[-1]",
        "A.x",
        "S[0]",
        "S.c",
        "A + 1.0",
        "-S",
        "float(A)",
        "max(A, A)",
        "true ? S : S",
    ],
)
def test_whole_aggregates_and_missing_elements_are_not_folded(expression):
    assert _fold(expression) is None


def test_member_access_can_yield_a_nested_aggregate():
    assert _fold("NS.inner") == NAMES["NS"].elements[0]


PROGRAM = textwrap.dedent("""
    shader S {
        struct P { float a; vec2 b; };
        const float A[3] = {1.0, 2.0, 3.0};
        const float B[3] = float[3](4.0, 5.0, 6.0);
        const float X = A[1] + B[2];
        const P S0 = P(1.5, vec2(2.0, 3.0));
        const float Y = S0.a;
        const vec2 Z = S0.b * 2.0;
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                data[0] = X + Y + Z.y + A[2] + B[0] + S0.b.x;
            }
        }
    }
    """)


def _raw_spirv(tmp_path, program):
    source = tmp_path / "constants.cgl"
    source.write_text(program, encoding="utf-8")
    # Unformatted output keeps numeric ids whether or not SPIRV-Tools is installed.
    return crosstl.translate(str(source), backend="vulkan", format_output=False)


def _value(raw, name_or_id):
    named = re.search(rf'^OpName %(\w+) "{name_or_id}"$', raw, re.M)
    identifier = named.group(1) if named else name_or_id
    definition = re.search(rf"^%{identifier} = (Op\w+) %\w+(.*)$", raw, re.M)
    assert definition, name_or_id
    opcode, operands = definition.group(1), definition.group(2).split()
    if opcode == "OpConstant":
        return float(operands[0])
    assert opcode == "OpConstantComposite", name_or_id
    return [_value(raw, operand.lstrip("%")) for operand in operands]


def test_spirv_aggregate_accesses_hold_the_folded_values(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    assert _value(raw, "A") == [1.0, 2.0, 3.0]
    assert _value(raw, "B") == [4.0, 5.0, 6.0]
    assert _value(raw, "X") == 8.0
    assert _value(raw, "S0") == [1.5, [2.0, 3.0]]
    assert _value(raw, "Y") == 1.5
    assert _value(raw, "Z") == [4.0, 6.0]


def test_spirv_module_scope_holds_no_access_instructions(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    module_scope = raw.split(" = OpFunction ", 1)[0]
    assert not re.search(
        r"= Op(F|I)(Add|Sub|Mul|Div) |= OpCompositeExtract |= OpCompositeConstruct "
        r"|= OpAccessChain |= OpVectorShuffle ",
        module_scope,
    )


def test_aggregate_initializers_without_a_value_are_refused(tmp_path):
    program = (
        "shader S { const float A[2] = {1.0, 2.0}; const float E = A[2];"
        " compute { void main() {} } }"
    )
    with pytest.raises(ValueError, match="E has an initializer that is not a constant"):
        _raw_spirv(tmp_path, program)


def test_spirv_aggregate_output_assembles_and_validates(tmp_path):
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

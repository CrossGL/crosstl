"""Module-scope matrix constants fold with column-major semantics.

The SPIR-V generator folded matrix constructors over scalars and vectors only.
For ``const vec2 C = M[1]``, ``const mat2 D = M * 2.0``, ``const vec2 P = M * v``
or ``const float E = M[0][1]`` it emitted the extract or the arithmetic at module
scope (invalid SPIR-V) and gave the constant the value zero.

Matrix initializers now fold column by column: ``m[i]`` is column ``i`` and
``m * v`` treats ``v`` as a column vector, matching ``OpMatrixTimesVector``.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.matrix_constant_folding import (
    MatrixConstant,
    MatrixConstantFolder,
)
from crosstl.translator.codegen.scalar_constant_folding import (
    ScalarConstant,
    fold_math_builtin,
)
from crosstl.translator.codegen.vector_constant_folding import VectorConstant

_KINDS = {"float", "double", "int", "uint", "bool"}


def _name(type_value):
    return (
        type_value if isinstance(type_value, str) else getattr(type_value, "name", None)
    )


def _kind(type_value):
    name = _name(type_value)
    return name if name in _KINDS else None


def _vector_shape(type_value):
    match = re.fullmatch(r"(i?)vec([234])", _name(type_value) or "")
    if not match:
        return None
    return ("int" if match.group(1) else "float"), int(match.group(2))


def _matrix_shape(type_value):
    match = re.fullmatch(r"mat([234])(?:x([234]))?", _name(type_value) or "")
    if not match:
        return None
    columns = int(match.group(1))
    return "float", columns, int(match.group(2) or columns)


def _vector(*values):
    return VectorConstant(
        "float", tuple(ScalarConstant("float", float(value)) for value in values)
    )


def _matrix(*columns):
    return MatrixConstant("float", tuple(_vector(*column) for column in columns))


M = _matrix((1, 2), (3, 4))
NAMES = {
    "M": M,
    "R": _matrix((1, 2, 3), (4, 5, 6)),  # mat2x3: two columns of three rows
    "V": _vector(1, 1),
}


def _fold(expression):
    program = parse(
        "shader S { const float probe = %s; compute { void main() {} } }" % expression
    )
    folder = MatrixConstantFolder(
        NAMES.get, _kind, _vector_shape, _matrix_shape, fold_builtin=fold_math_builtin
    )
    return folder.fold(program.constants[0].value)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("mat2(1.0, 2.0, 3.0, 4.0)", M),
        ("mat2(vec2(1.0, 2.0), vec2(3.0, 4.0))", M),
        ("mat2(2.0)", _matrix((2, 0), (0, 2))),
        ("mat3(M)", _matrix((1, 2, 0), (3, 4, 0), (0, 0, 1))),
        ("mat2(R)", _matrix((1, 2), (4, 5))),
        ("M[1]", _vector(3, 4)),
        ("M[0][1]", ScalarConstant("float", 2.0)),
        ("M[1].yx", _vector(4, 3)),
        ("-M", _matrix((-1, -2), (-3, -4))),
        ("M * 2.0", _matrix((2, 4), (6, 8))),
        ("1.0 - M", _matrix((0, -1), (-2, -3))),
        ("M + M", _matrix((2, 4), (6, 8))),
        ("M / M", _matrix((1, 1), (1, 1))),
        # Column vector: (1*1 + 3*1, 2*1 + 4*1).
        ("M * V", _vector(4, 6)),
        # Row vector: (dot(V, column 0), dot(V, column 1)).
        ("V * M", _vector(3, 7)),
        ("M * M", _matrix((7, 10), (15, 22))),
        # mat2x3 * vec2 is a vec3; vec3 * mat2x3 is a vec2.
        ("R * V", _vector(5, 7, 9)),
        ("vec3(1.0, 0.0, 1.0) * R", _vector(4, 10)),
        ("transpose(M)", _matrix((1, 3), (2, 4))),
        ("transpose(R)", _matrix((1, 4), (2, 5), (3, 6))),
        ("matrixCompMult(M, M)", _matrix((1, 4), (9, 16))),
        (
            "outerProduct(vec2(1.0, 2.0), vec3(1.0, 2.0, 3.0))",
            _matrix((1, 2), (2, 4), (3, 6)),
        ),
        ("true ? M : mat2(0.0)", M),
    ],
)
def test_matrix_expressions_fold_column_major(expression, expected):
    assert _fold(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "M == M",
        "M[2]",
        "M + R",
        "M * R",
        "R * R",
        "M * vec3(1.0)",
        "vec3(1.0) * M",
        "mat2(1.0, 2.0, 3.0)",
        "mat2(M, 1.0)",
        "determinant(M)",
        "inverse(M)",
        "M.x",
        "M % M",
        "sqrt(M)",
        "vec4(M)",
        "transpose(V)",
        "M ? M : M",
    ],
)
def test_matrix_expressions_without_a_folded_value_are_not_folded(expression):
    assert _fold(expression) is None


PROGRAM = textwrap.dedent("""
    shader S {
        const mat2 M = mat2(1.0, 2.0, 3.0, 4.0);
        const vec2 C = M[1];
        const mat2 D = M * 2.0;
        const vec2 P = M * vec2(1.0, 1.0);
        const vec2 Q = vec2(1.0, 1.0) * M;
        const float E = M[0][1];
        const mat2 T = transpose(M) * M;
        const mat3 X = mat3(M);
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                data[0] = C.x + D[0][0] + P.y + Q.x + E + T[1][1] + X[2][2];
            }
        }
    }
    """)


def _raw_spirv(tmp_path, program):
    source = tmp_path / "constants.cgl"
    source.write_text(program, encoding="utf-8")
    # Unformatted output keeps numeric ids whether or not SPIRV-Tools is installed.
    return crosstl.translate(str(source), backend="vulkan", format_output=False)


def _definition(raw, name_or_id):
    named = re.search(rf'^OpName %(\w+) "{name_or_id}"$', raw, re.M)
    identifier = named.group(1) if named else name_or_id
    definition = re.search(rf"^%{identifier} = (Op\w+) %\w+(.*)$", raw, re.M)
    assert definition, name_or_id
    return definition.group(1), definition.group(2).split()


def _value(raw, name_or_id):
    opcode, operands = _definition(raw, name_or_id)
    if opcode == "OpConstant":
        return float(operands[0])
    assert opcode == "OpConstantComposite", name_or_id
    return [_value(raw, operand.lstrip("%")) for operand in operands]


def test_spirv_matrix_constants_hold_the_folded_values(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    assert _value(raw, "M") == [[1.0, 2.0], [3.0, 4.0]]
    assert _value(raw, "C") == [3.0, 4.0]
    assert _value(raw, "D") == [[2.0, 4.0], [6.0, 8.0]]
    assert _value(raw, "P") == [4.0, 6.0]
    assert _value(raw, "Q") == [3.0, 7.0]
    assert _value(raw, "E") == 2.0
    # transpose(M) * M = [[1, 2], [3, 4]]^T [[1, 2], [3, 4]] by columns.
    assert _value(raw, "T") == [[5.0, 11.0], [11.0, 25.0]]
    assert _value(raw, "X") == [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]


def test_spirv_module_scope_holds_no_matrix_instructions(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    module_scope = raw.split(" = OpFunction ", 1)[0]
    assert not re.search(
        r"= Op(F|I)(Add|Sub|Mul|Div|Negate) |= Op(Matrix|Vector)Times"
        r"|= OpTranspose |= OpCompositeExtract |= OpCompositeConstruct ",
        module_scope,
    )


def test_matrix_initializers_without_a_value_are_refused(tmp_path):
    program = (
        "shader S { const mat2 M = mat2(1.0); const float E = determinant(M);"
        " compute { void main() {} } }"
    )
    with pytest.raises(ValueError, match="E has an initializer that is not a constant"):
        _raw_spirv(tmp_path, program)


def test_spirv_matrix_output_assembles_and_validates(tmp_path):
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

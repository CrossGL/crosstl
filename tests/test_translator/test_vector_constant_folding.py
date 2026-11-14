"""Module-scope vector constants fold component by component.

The SPIR-V generator only folded vector constructors over literals. For
``const vec2 W = V * 2.0``, ``const vec2 N = -V``, ``const float X = V.y`` or
``const vec3 Q = vec3(V.yx, 1.0)`` it emitted the arithmetic, a swizzle or a
shuffle at module scope (invalid SPIR-V) and gave the constant the value zero.

Vector initializers now fold with the scalar folder's arithmetic for each
component, and initializers that cannot be folded are refused.
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
    fold_math_builtin,
    round_to_binary32,
)
from crosstl.translator.codegen.vector_constant_folding import (
    VectorConstant,
    VectorConstantFolder,
    swizzle_indices,
)

_KINDS = {"float", "double", "int", "uint", "bool"}
_VECTOR_PREFIXES = {"vec": "float", "ivec": "int", "uvec": "uint", "bvec": "bool"}


def _name(type_value):
    return (
        type_value if isinstance(type_value, str) else getattr(type_value, "name", None)
    )


def _kind(type_value):
    name = _name(type_value)
    return name if name in _KINDS else None


def _shape(type_value):
    name = _name(type_value)
    match = re.fullmatch(r"([a-z]*vec)([234])", name or "")
    if not match or match.group(1) not in _VECTOR_PREFIXES:
        return None
    return _VECTOR_PREFIXES[match.group(1)], int(match.group(2))


def _vector(kind, *values):
    return VectorConstant(kind, tuple(ScalarConstant(kind, value) for value in values))


V = _vector("float", 1.0, 2.0)
NAMES = {
    "V": V,
    "C": _vector("float", 1.0, 2.0, 3.0),
    "A": ScalarConstant("float", 0.1),
}


def _fold(expression):
    program = parse(
        "shader S { const float probe = %s; compute { void main() {} } }" % expression
    )
    folder = VectorConstantFolder(
        NAMES.get, _kind, _shape, fold_builtin=fold_math_builtin
    )
    return folder.fold(program.constants[0].value)


def _f(value):
    return round_to_binary32(value)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("V.y", ScalarConstant("float", 2.0)),
        ("V[1]", ScalarConstant("float", 2.0)),
        ("C.zyx", _vector("float", 3.0, 2.0, 1.0)),
        ("C.bg", _vector("float", 3.0, 2.0)),
        ("V * 2.0", _vector("float", 2.0, 4.0)),
        ("2.0 - V", _vector("float", 1.0, 0.0)),
        ("-V", _vector("float", -1.0, -2.0)),
        ("V / vec2(0.0, 4.0)", _vector("float", math.inf, 0.5)),
        ("V * A", _vector("float", _f(0.1), _f(_f(0.1) * 2.0))),
        ("vec3(V.yx, 1.0)", _vector("float", 2.0, 1.0, 1.0)),
        ("vec4(V, V)", _vector("float", 1.0, 2.0, 1.0, 2.0)),
        ("vec2(C)", _vector("float", 1.0, 2.0)),
        ("ivec2(3)", _vector("int", 3, 3)),
        ("ivec2(-2.5, 7.9)", _vector("int", -2, 7)),
        ("uvec2(1u, 2u) - uvec2(2u)", _vector("uint", (1 << 32) - 1, 0)),
        ("ivec2(5, -5) / 2", _vector("int", 2, -2)),
        ("ivec2(1, 2) << 3", _vector("int", 8, 16)),
        ("~ivec2(0, 1)", _vector("int", -1, -2)),
        ("bvec2(true, false)", _vector("bool", True, False)),
        ("true ? V : vec2(0.0)", V),
        ("vec2(1, 2) + ivec2(1, 1)", _vector("float", 2.0, 3.0)),
        ("clamp(C, 1.5, 2.5)", _vector("float", 1.5, 2.0, 2.5)),
        ("max(V, vec2(1.5))", _vector("float", 1.5, 2.0)),
        ("sqrt(V)", _vector("float", 1.0, _f(math.sqrt(2.0)))),
        ("mix(V, vec2(3.0), 0.5)", _vector("float", 2.0, 2.5)),
        ("dot(C, C)", ScalarConstant("float", 14.0)),
        ("length(vec2(3.0, 4.0))", ScalarConstant("float", 5.0)),
        ("distance(V, vec2(4.0, 6.0))", ScalarConstant("float", 5.0)),
        ("normalize(vec2(3.0, 4.0))", _vector("float", _f(0.6), _f(0.8))),
        (
            "cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))",
            _vector("float", 0.0, 0.0, 1.0),
        ),
        ("any(bvec2(false, true))", ScalarConstant("bool", True)),
        ("all(bvec2(false, true))", ScalarConstant("bool", False)),
        ("not(bvec2(false, true))", _vector("bool", True, False)),
        ("float(C)", ScalarConstant("float", 1.0)),
    ],
)
def test_vector_expressions_fold_component_by_component(expression, expected):
    assert _fold(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        # GLSL compares whole vectors, HLSL compares components.
        "V == V",
        "V < vec2(3.0)",
        "V && V",
        "V.z",
        "V.xr",
        "V[2]",
        "V + C",
        "vec3(V)",
        "vec3(1.0, 2.0)",
        "vec2(1.0, 2.0, 3.0)",
        "vec2(C, 1.0)",
        "V ? V : V",
        "normalize(vec2(0.0))",
        "cross(V, V)",
        "ivec2(1, 2) / 0",
        "sqrt(-V)",
        "any(V)",
        "unknown(V)",
    ],
)
def test_vector_expressions_without_a_defined_value_are_not_folded(expression):
    assert _fold(expression) is None


def test_swizzle_indices_keep_to_one_letter_set_and_the_vector_size():
    assert swizzle_indices("wzyx", 4) == (3, 2, 1, 0)
    assert swizzle_indices("st", 2) == (0, 1)
    assert swizzle_indices("xg", 4) is None
    assert swizzle_indices("z", 2) is None
    assert swizzle_indices("xxxxx", 4) is None


PROGRAM = textwrap.dedent("""
    shader S {
        const float A = 0.1;
        const vec2 V = vec2(1.0, 2.0);
        const float X = V.y;
        const vec2 W = V * 2.0;
        const vec2 N = -V;
        const vec3 Q = vec3(V.yx, A);
        const ivec2 I = ivec2(7, -7) / 2;
        const float L = length(vec2(3.0, 4.0));
        const vec2 H = normalize(W);
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                data[0] = X + W.x + N.y + Q.z + float(I.y) + L + H.x;
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


def _components(raw, name):
    opcode, components = _definition(raw, name)
    assert opcode == "OpConstantComposite", name
    values = []
    for component in components:
        match = re.search(rf"^{component} = OpConstant %\w+ (\S+)$", raw, re.M)
        assert match, (name, component)
        values.append(match.group(1))
    return values


def test_spirv_vector_constants_hold_the_folded_values(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    assert _definition(raw, "X") == ("OpConstant", ["2.0"])
    assert _definition(raw, "L") == ("OpConstant", ["5.0"])
    assert _components(raw, "W") == ["2.0", "4.0"]
    assert _components(raw, "N") == ["-1.0", "-2.0"]
    assert _components(raw, "Q") == ["2.0", "1.0", "0.1"]
    assert _components(raw, "I") == ["3", "-3"]
    assert _components(raw, "H") == ["0.4472136", "0.8944272"]


def test_spirv_module_scope_holds_no_vector_instructions(tmp_path):
    raw = _raw_spirv(tmp_path, PROGRAM)
    module_scope = raw.split(" = OpFunction ", 1)[0]
    assert not re.search(
        r"= Op(F|I|S|U)(Add|Sub|Mul|Div|Negate)|= OpExtInst |= OpVectorShuffle"
        r"|= OpCompositeExtract |= OpCompositeConstruct ",
        module_scope,
    )


def test_vector_initializers_without_a_value_are_refused(tmp_path):
    program = (
        "shader S { const vec2 V = vec2(1.0, 2.0); const bool E = V == V;"
        " compute { void main() {} } }"
    )
    with pytest.raises(ValueError, match="E has an initializer that is not a constant"):
        _raw_spirv(tmp_path, program)


def test_spirv_vector_output_assembles_and_validates(tmp_path):
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

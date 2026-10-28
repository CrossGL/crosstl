"""Structure constructor arguments take their members' types, and CUDA and HIP build structures by aggregate."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.struct_constructor_conversions import (
    convert_struct_constructor_arguments,
)

ALL_TARGETS = [
    "metal",
    "directx",
    "opengl",
    "vulkan",
    "wgsl",
    "slang",
    "cuda",
    "hip",
    "rust",
    "mojo",
]

# Constructors whose arguments are of the member types, nested, assigned, returned and used as a value.
PAIRS = """
shader StructConstructors {
    struct Pair {
        float a;
        float b;
    };
    struct Outer {
        Pair p;
        vec3 v;
        int k;
    };
    Pair makePair(float x, int k) {
        return Pair(x, k);
    }
    float use(float x, int k, uint u) {
        Pair p = Pair(x, k);
        Outer o = Outer(Pair(1.0, x), vec3(1.0), 3);
        p = Pair(p.b, u);
        Pair q = makePair(x, k);
        return p.a + p.b + o.p.b + o.v.x + float(o.k) + q.b + Pair(4.0, 5.0).a;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            float r = use(1.5, 2, 3u);
        }
    }
}
"""

# Vector, unsigned and boolean conversions, and a program constant initializer.
CELLS = """
shader VectorMembers {
    struct Cell {
        vec3 position;
        uint id;
        float weight;
    };
    const Cell ORIGIN = Cell(vec3(0.0), 0, 1);
    float use(ivec3 p, int id, bool on) {
        Cell c = Cell(p, id, on);
        return c.position.x + float(c.id) + c.weight + ORIGIN.weight;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            float r = use(ivec3(1, 2, 3), 4, true);
        }
    }
}
"""


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _function(program, name):
    return next(function for function in program.functions if function.name == name)


def _call_text(expr):
    """A compact CrossGL spelling of an expression made of calls, identifiers and literals."""
    if isinstance(expr, A.FunctionCallNode):
        arguments = ", ".join(_call_text(argument) for argument in expr.arguments)
        return f"{expr.function.name}({arguments})"
    if isinstance(expr, A.IdentifierNode):
        return expr.name
    if isinstance(expr, A.LiteralNode):
        return str(expr.value)
    if isinstance(expr, A.MemberAccessNode):
        return f"{_call_text(expr.object)}.{expr.member}"
    return type(expr).__name__


def _initializers(function):
    return [
        _call_text(statement.initial_value)
        for statement in function.body.statements
        if isinstance(statement, A.VariableNode) and statement.initial_value
    ]


def _program_body(text):
    """The part of a CUDA or HIP translation after its matrix and vector helper prelude."""
    marker = text.find("struct Pair")
    if marker < 0:
        marker = text.find("struct Cell")
    return text[marker:] if marker >= 0 else text


def test_arguments_of_other_types_are_converted_to_the_member_types():
    program = _parse(PAIRS)
    assert convert_struct_constructor_arguments(program) == 3
    assert _initializers(_function(program, "use"))[:2] == [
        "Pair(x, float(k))",
        "Outer(Pair(1.0, x), vec3(1.0), 3)",
    ]
    assignment = _function(program, "use").body.statements[2].expression
    assert _call_text(assignment.value) == "Pair(p.b, float(u))"
    returned = _function(program, "makePair").body.statements[0]
    assert _call_text(returned.value) == "Pair(x, float(k))"


def test_vector_unsigned_boolean_and_constant_arguments_are_converted():
    program = _parse(CELLS)
    assert convert_struct_constructor_arguments(program) == 5
    assert _initializers(_function(program, "use"))[0] == (
        "Cell(vec3(p), uint(id), float(on))"
    )
    assert _call_text(program.constants[0].value) == (
        "Cell(vec3(0.0), uint(0), float(1))"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        # Already of the member types.
        "Pair(x, 2.0)",
        # Fewer arguments than members: no member-by-member reading applies.
        "Pair(x)",
        # A different number of lanes is not an implicit conversion.
        "Pair(x, vec2(1.0))",
        # The argument's type is unknown.
        "Pair(x, unknownValue)",
    ],
)
def test_other_constructors_are_left_as_they_are(arguments):
    program = _parse(f"""
shader S {{
    struct Pair {{ float a; float b; }};
    float use(float x) {{
        Pair p = {arguments};
        return p.a;
    }}
}}
""")
    assert convert_struct_constructor_arguments(program) == 0
    assert _initializers(_function(program, "use")) == [arguments]


def test_structure_and_array_members_are_not_converted():
    program = _parse("""
shader S {
    struct Inner { float a; };
    struct Holder { Inner inner; float values[2]; };
    float use(Inner i, float v[2]) {
        Holder h = Holder(i, v);
        return h.inner.a;
    }
}
""")
    assert convert_struct_constructor_arguments(program) == 0


@pytest.mark.parametrize("target", ALL_TARGETS)
@pytest.mark.parametrize("source", [PAIRS, CELLS], ids=["pairs", "cells"])
def test_every_target_translates(tmp_path, target, source):
    assert crosstl.translate(_write(tmp_path, "s.cgl", source), backend=target)


@pytest.mark.parametrize("target", ["cuda", "hip"])
def test_cuda_and_hip_build_structures_by_aggregate_initialization(tmp_path, target):
    output = _program_body(
        crosstl.translate(_write(tmp_path, "s.cgl", PAIRS), backend=target)
    )
    # The structures declare no constructor, so a call form has nothing to call.
    assert re.search(r"\b(Pair|Outer)\(", output) is None
    assert "return Pair{x, float(k)};" in output
    assert "Pair p = Pair{x, float(k)};" in output
    assert "Outer{Pair{1.0f, x}, make_float3(1.0f, 1.0f, 1.0f), 3}" in output
    assert "Pair{4.0f, 5.0f}.a" in output
    assert "makePair(x, k)" in output


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("metal", "Pair p = Pair{x, float(k)};"),
        ("wgsl", "var p: Pair = Pair(x, f32(k));"),
        ("mojo", "var p: Pair = Pair(x, Float32(k))"),
    ],
)
def test_targets_without_implicit_conversions_write_them(tmp_path, target, expected):
    output = crosstl.translate(_write(tmp_path, "s.cgl", PAIRS), backend=target)
    assert expected in output


def test_metal_cells_convert_vector_unsigned_and_boolean_members(tmp_path):
    output = crosstl.translate(_write(tmp_path, "s.cgl", CELLS), backend="metal")
    assert "Cell{float3(p), uint(id), float(on)}" in output


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", [PAIRS, CELLS], ids=["pairs", "cells"])
def test_wgsl_output_validates_with_naga(tmp_path, source):
    if shutil.which("naga") is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "s.wgsl"
    path.write_text(
        crosstl.translate(_write(tmp_path, "s.cgl", source), backend="wgsl")
    )
    _run(["naga", str(path)])


@pytest.mark.parametrize("source", [PAIRS, CELLS], ids=["pairs", "cells"])
def test_metal_output_compiles(tmp_path, source):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal compiler is not installed")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not installed")
    path = tmp_path / "s.metal"
    path.write_text(
        crosstl.translate(_write(tmp_path, "s.cgl", source), backend="metal")
    )
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "s.air"),
        ]
    )


def test_cuda_output_compiles_with_nvcc(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    path = tmp_path / "s.cu"
    path.write_text(crosstl.translate(_write(tmp_path, "s.cgl", PAIRS), backend="cuda"))
    _run([nvcc, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "s.o")])


def test_hip_output_compiles_with_hipcc(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    path = tmp_path / "s.hip"
    path.write_text(crosstl.translate(_write(tmp_path, "s.cgl", PAIRS), backend="hip"))
    _run(
        [
            hipcc,
            "-std=c++17",
            "-Werror=c++11-narrowing",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "s.o"),
        ]
    )

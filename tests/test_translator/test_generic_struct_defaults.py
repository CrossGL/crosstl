"""Default arguments of generic structures are spelled out at every use.

``generic<T, U = vec2> struct Pair`` used as ``Pair<vec2>`` matched no specialization: GLSL, SPIR-V and WGSL refused
the program, and Metal and HLSL emitted ``Pair<float2>`` verbatim, naming a template the output never declares. Each
case below fails on that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.generic_struct_defaults import (
    GenericStructArityError,
    fill_generic_struct_defaults,
)

PROGRAM = textwrap.dedent("""
    shader main {
        generic<T, U = vec2> struct Pair {
            T first;
            U second;
        }
        generic<T = float> struct Cell {
            T v;
        }
        generic<T, U = T> struct Twin {
            T a;
            U b;
        }
        compute {
            RWStructuredBuffer<float> f @buffer(0);
            @numthreads(1, 1, 1)
            void main(uvec3 id @ gl_GlobalInvocationID) {
                Pair<vec2> p;
                p.first = vec2(f[1], 1.0);
                p.second = p.first;
                Pair<float, int> q;
                q.first = f[2];
                q.second = 3;
                Cell<> c;
                c.v = f[3];
                Twin<int> t;
                t.a = 1;
                t.b = 2;
                f[0] = p.second.x + q.first + float(q.second) + c.v + float(t.a + t.b);
            }
        }
    }
""").strip()


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools are not installed")
        path, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        path.write_text(output)
        subprocess.run(
            [assembler, "--target-env", "vulkan1.1", str(path), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        command = [validator, "--target-env", "vulkan1.1", str(binary)]
    elif backend == "wgsl":
        tool = shutil.which("naga")
        if tool is None:
            pytest.skip("naga is not installed")
        path = tmp_path / "out.wgsl"
        path.write_text(output)
        command = [tool, str(path)]
    elif backend == "metal":
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "out.metal"
        path.write_text(output)
        command = [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"]
    else:
        tool = shutil.which("dxc")
        if tool is None:
            pytest.skip("dxc is not installed")
        path = tmp_path / "out.hlsl"
        path.write_text(output)
        command = [tool, "-HV", "2021", "-T", "cs_6_2", "-E", "CSMain", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan", "wgsl"])
def test_uses_that_leave_defaults_out_translate(tmp_path, backend):
    path = tmp_path / "defaults.cgl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend=backend)

    assert "Pair<" not in output and "Cell<" not in output and "Twin<" not in output
    _validate(tmp_path, backend, output)


@pytest.mark.parametrize(
    ("backend", "declarations"),
    [
        ("metal", ("Pair_float2_float2 p;", "Cell_float c;", "Twin_int_int t;")),
        ("opengl", ("Pair_vec2_vec2 p;", "Cell_float c;", "Twin_int_int t;")),
    ],
)
def test_defaults_name_the_full_specialization(tmp_path, backend, declarations):
    path = tmp_path / "defaults.cgl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend=backend)

    for declaration in declarations:
        assert declaration in output, declaration


def _variable_types(program):
    entry = next(iter(program.stages.values())).entry_point
    return {
        statement.name: statement.var_type
        for statement in entry.body.statements
        if isinstance(statement, A.VariableNode)
    }


def test_the_pass_completes_types_and_constructor_callees():
    program = Parser(
        Lexer(
            PROGRAM.replace("Pair<vec2> p;", "Pair<vec2> p = Pair<vec2>(vec2(0.0));")
        ).get_tokens()
    ).parse()

    # Three variable types and one constructor callee.
    assert fill_generic_struct_defaults(program) == 4
    types = _variable_types(program)
    assert [repr(arg) for arg in types["p"].generic_args] == [
        repr(A.VectorType(A.PrimitiveType("float"), 2))
    ] * 2
    assert [repr(arg) for arg in types["c"].generic_args] == [
        repr(A.PrimitiveType("float"))
    ]
    # ``U = T`` takes the argument given for ``T``.
    assert [arg.name for arg in types["t"].generic_args] == ["int", "int"]
    entry = next(iter(program.stages.values())).entry_point
    initializer = entry.body.statements[0].initial_value
    assert initializer.function.name == "Pair<vec2, vec2>"


def test_constructor_callee_defaults_take_earlier_arguments():
    source = PROGRAM.replace(
        "generic<T = float> struct Cell {",
        "generic<T, U = vec2<T>> struct Lane {\n            T a;\n            U b;\n        }\n"
        "        generic<T = float> struct Cell {",
    ).replace(
        "Twin<int> t;",
        "Twin<int> t = Twin<int>(1, 2);\n                Lane<int> l = Lane<int>(1, ivec2(2));",
    )
    program = Parser(Lexer(source).get_tokens()).parse()

    fill_generic_struct_defaults(program)

    entry = next(iter(program.stages.values())).entry_point
    callees = {
        statement.name: statement.initial_value.function.name
        for statement in entry.body.statements
        if isinstance(statement, A.VariableNode)
        and isinstance(statement.initial_value, A.FunctionCallNode)
    }
    # ``U = T`` and ``U = vec2<T>`` both take the argument given for ``T``.
    assert callees == {"t": "Twin<int, int>", "l": "Lane<int, vec2<int>>"}


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx"])
@pytest.mark.parametrize(
    ("use", "given"), [("Pair<float> p;", 1), ("Pair<float, int, int> p;", 3)]
)
def test_uses_of_the_wrong_arity_are_refused(tmp_path, backend, use, given):
    path = tmp_path / "arity.cgl"
    path.write_text(textwrap.dedent(f"""
            shader main {{
                generic<T, U> struct Pair {{
                    T first;
                    U second;
                }}
                compute {{
                    RWStructuredBuffer<float> f @buffer(0);
                    @numthreads(1, 1, 1)
                    void main(uvec3 id @ gl_GlobalInvocationID) {{
                        {use}
                        p.first = f[1];
                        f[0] = p.first;
                    }}
                }}
            }}
        """))

    # Metal, HLSL and GLSL emitted the use verbatim, naming a template the output never declares.
    with pytest.raises(
        GenericStructArityError,
        match=rf"'Pair' takes 2 arguments, but a use gives {given}",
    ):
        crosstl.translate(str(path), backend=backend)

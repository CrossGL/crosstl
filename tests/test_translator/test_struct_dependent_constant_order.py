"""Constants that read a struct-typed constant are declared after it.

GLSL, HLSL, Metal and Slang output declares constants of plain types before the
struct declarations and struct-typed constants after them. A plain-typed
constant whose initializer read a struct constant (``const float Y = S0.a``)
or called a struct constructor stayed in the first group, so it referenced an
identifier that was not declared yet and the output did not compile.

Such constants, and every constant that depends on them, now follow the
structs, keeping source order within each group.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.constant_ordering import (
    expression_reference_names,
    partition_constants_by_struct_dependency,
)

PROGRAM = textwrap.dedent("""
    shader S {
        const int N = 2;
        struct Q { float w[N]; };
        struct P { float a; vec2 b; };
        const float A[3] = {1.0, 2.0, 3.0};
        const float X = A[1];
        const P S0 = P(1.0, vec2(2.0, 3.0));
        const float Y = S0.a;
        const vec2 Z = S0.b * 2.0;
        const float W = Y * 2.0;
        const float V = P(2.0, vec2(0.0)).a;
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                data[0] = X + Y + Z.y + W + V + S0.b.x;
            }
        }
    }
    """)


def test_constants_reading_struct_constants_follow_the_structs():
    program = parse(PROGRAM)
    leading, dependent = partition_constants_by_struct_dependency(
        program.constants, program.structs
    )
    assert [c.name for c in leading] == ["N", "A", "X"]
    assert [c.name for c in dependent] == ["S0", "Y", "Z", "W", "V"]


def test_member_names_are_not_references():
    program = parse(
        "shader S { const float probe = s.a + v.xy.x + t[i]; compute { void main() {} } }"
    )
    assert expression_reference_names(program.constants[0].value) == {
        "s",
        "v",
        "t",
        "i",
    }


def _declaration_line(code, name):
    match = re.search(rf"^[^=\n]*\b{name}(\[[^\]]*\])? =", code, re.M)
    assert match, name
    return code.count("\n", 0, match.start())


def _struct_line(code, name="P"):
    match = re.search(rf"^\s*struct {name}\b", code, re.M)
    assert match
    return code.count("\n", 0, match.start())


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal", "slang"])
def test_declarations_follow_their_dependencies(tmp_path, backend):
    source = tmp_path / "order.cgl"
    source.write_text(PROGRAM, encoding="utf-8")
    code = crosstl.translate(str(source), backend=backend)
    struct = _struct_line(code)
    assert _declaration_line(code, "N") < _struct_line(code, "Q")
    assert _declaration_line(code, "X") < struct
    s0 = _declaration_line(code, "S0")
    assert struct < s0
    for name in ("Y", "Z", "W", "V"):
        assert s0 < _declaration_line(code, name), name


_COMPILERS = {
    "opengl": (
        "glslangValidator",
        "order.comp",
        lambda tool: [tool, "-S", "comp", "order.comp"],
    ),
    "metal": (
        "xcrun",
        "order.metal",
        lambda tool: [
            tool,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "order.metal",
            "-o",
            "o.air",
        ],
    ),
    "directx": (
        "dxc",
        "order.hlsl",
        lambda tool: [tool, "-T", "cs_6_0", "-E", "CSMain", "order.hlsl"],
    ),
}


@pytest.mark.parametrize("backend", sorted(_COMPILERS))
def test_output_compiles_natively(tmp_path, backend):
    tool_name, file_name, command = _COMPILERS[backend]
    tool = shutil.which(tool_name)
    if tool is None:
        pytest.skip(f"{tool_name} is not installed")
    source = tmp_path / "order.cgl"
    source.write_text(PROGRAM, encoding="utf-8")
    (tmp_path / file_name).write_text(
        crosstl.translate(str(source), backend=backend), encoding="utf-8"
    )
    result = subprocess.run(
        command(tool), capture_output=True, text=True, timeout=300, cwd=tmp_path
    )
    if backend == "metal" and "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]

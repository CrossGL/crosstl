"""Struct constants with array members compile on HLSL and Metal.

HLSL emits struct constructors as calls to a generated helper. For
``const P S1 = P(2.0, {6.0, 7.0})`` it passed the array literal ``{6.0, 7.0}``
as a call argument, which is not HLSL. Metal emitted ``P{1.0, WS}`` for a
named array constant ``WS``, and C++ cannot copy-initialize an array member
from an lvalue array.

Changes:

* HLSL declaration initializers now use an initializer list
  (``{2.0, {6.0, 7.0}}``).
* Other HLSL expression contexts are refused with a typed diagnostic.
* Metal types program-scope constants, so an array constant copied into an
  array member is expanded element by element (``{WS[0], WS[1]}``).
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader S {
        struct P { float a; float w[2]; };
        struct Q { P p; float b; };
        const float WS[2] = {4.0, 5.0};
        const P S0 = P(1.0, WS);
        const P S1 = P(2.0, {6.0, 7.0});
        const Q Q0 = Q(P(1.0, {2.0, 3.0}), 4.0);
        compute {
            buffer float data[];
            layout(local_size_x = 1) in;
            void main() {
                P local = P(3.0, WS);
                P local2 = P(3.0, {8.0, 9.0});
                data[0] = S0.w[1] + S1.w[0] + Q0.p.w[1] + local.w[1] + local2.w[0];
            }
        }
    }
    """)


def _translate(tmp_path, program, backend):
    source = tmp_path / "members.cgl"
    source.write_text(program, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend)


def test_hlsl_struct_constants_with_array_literals_use_initializer_lists(tmp_path):
    code = _translate(tmp_path, PROGRAM, "directx")
    assert "static const P S1 = {2.0, {6.0, 7.0}};" in code
    assert "static const Q Q0 = {{1.0, {2.0, 3.0}}, 4.0};" in code
    # A named array is a valid helper argument and keeps the helper form.
    assert "static const P S0 = __crossgl_construct_P(1.0, WS);" in code


def test_hlsl_array_literal_fields_outside_initializers_are_refused(tmp_path):
    program = PROGRAM.replace(
        "const Q Q0 = Q(P(1.0, {2.0, 3.0}), 4.0);",
        "const Q Q0 = Q(P(1.0, {2.0, 3.0}), 4.0);\n"
        "const float V = P(2.0, {0.0, 1.0}).w[1];",
    )
    with pytest.raises(ValueError, match="no fixed-array literal expression for a P"):
        _translate(tmp_path, program, "directx")


def test_metal_array_constants_are_copied_element_by_element(tmp_path):
    code = _translate(tmp_path, PROGRAM, "metal")
    assert "constant P S0 = P{1.0, {WS[0], WS[1]}};" in code
    assert "P local = P{3.0, {WS[0], WS[1]}};" in code


_COMPILERS = {
    "opengl": (
        "glslangValidator",
        "members.comp",
        lambda tool: [tool, "-S", "comp", "members.comp"],
    ),
    "metal": (
        "xcrun",
        "members.metal",
        lambda tool: [
            tool,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "members.metal",
            "-o",
            "m.air",
        ],
    ),
    "directx": (
        "dxc",
        "members.hlsl",
        lambda tool: [tool, "-T", "cs_6_0", "-E", "CSMain", "members.hlsl"],
    ),
}


@pytest.mark.parametrize("backend", sorted(_COMPILERS))
def test_output_compiles_natively(tmp_path, backend):
    tool_name, file_name, command = _COMPILERS[backend]
    tool = shutil.which(tool_name)
    if tool is None:
        pytest.skip(f"{tool_name} is not installed")
    (tmp_path / file_name).write_text(
        _translate(tmp_path, PROGRAM, backend), encoding="utf-8"
    )
    result = subprocess.run(
        command(tool), capture_output=True, text=True, timeout=300, cwd=tmp_path
    )
    if backend == "metal" and "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]

"""Array constructors with several dimensions or a size that names a constant (#3679, #3553).

Metal, HLSL and the SPIR-V generator build an array only in an aggregate initializer, so
``crosstl/translator/semantics/array_values.py`` rewrites each GLSL array constructor into an initializer list. It
recognised only ``element[literal](...)``: ``float[2][2](float[2](...), ...)``, which the parser reads as the type
``float`` indexed twice, and ``float[N](...)`` reached Metal and HLSL output verbatim, which neither compiler accepts,
and the SPIR-V generator refused them as calls through a callee expression.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics.array_values import array_constructor_shape

_PROGRAM = """
shader P {{
    {scope}
    buffer float* outp;
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""

_NESTED = "float[2][2](float[2](1.0, 2.0), float[2](3.0, 4.0))"

# (shader-scope declarations, entry body, the Metal and HLSL text each must contain)
_CASES = {
    "nested-declaration": (
        "",
        f"float L[2][2] = {_NESTED}; outp[i] = L[1][0];",
        "float L[2][2] = {{1.0, 2.0}, {3.0, 4.0}};",
    ),
    "nested-assignment": (
        "",
        f"float L[2][2]; L = {_NESTED}; outp[i] = L[1][0];",
        "cgl_array_value_0[2][2] = {{1.0, 2.0}, {3.0, 4.0}};",
    ),
    "two-by-three": (
        "",
        "float L[2][3] = float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0)); outp[i] = L[1][2];",
        "float L[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};",
    ),
    "named-size-declaration": (
        "const int N = 3;",
        "float a[N] = float[N](1.0, 2.0, 3.0); outp[i] = a[N - 1];",
        "a[N] = {1.0, 2.0, 3.0};",
    ),
    "named-size-argument": (
        "const int N = 2; float total(float v[2]) { return v[0] + v[1]; }",
        "outp[i] = total(float[N](1.0, 2.0));",
        "cgl_array_value_0[2] = {1.0, 2.0};",
    ),
    "named-inner-size": (
        "const int N = 2;",
        "float L[2][N] = float[2][N](float[N](1.0, 2.0), float[N](3.0, 4.0)); outp[i] = L[1][0];",
        "{{1.0, 2.0}, {3.0, 4.0}};",
    ),
}


def _translate(tmp_path, backend: str, scope: str, body: str) -> str:
    path = tmp_path / "program.cgl"
    path.write_text(_PROGRAM.format(scope=scope, body=body))
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", ["metal", "directx"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_array_constructors_become_initializer_lists(tmp_path, backend, case):
    scope, body, expected = _CASES[case]
    written = _translate(tmp_path, backend, scope, body)
    assert expected in written
    assert "float[2](" not in written and "float[N](" not in written


def test_a_renamed_size_constant_keeps_its_declaration_and_list(tmp_path):
    # Metal renames a constant named like a Metal function (``using namespace metal`` declares ``level``); the
    # constructor's own spelling is gone, so it cannot keep the old name.
    written = _translate(
        tmp_path,
        "metal",
        "const int level = 2;",
        "float w[level] = float[level](1.0, 2.0); outp[i] = w[1];",
    )
    assert "float w[level_] = {1.0, 2.0};" in written


def _indexed(base: str, *sizes) -> A.ASTNode:
    callee = A.IdentifierNode(base)
    for size in sizes:
        index = (
            A.LiteralNode(size, A.PrimitiveType("int"))
            if isinstance(size, int)
            else A.IdentifierNode(size)
        )
        callee = A.ArrayAccessNode(callee, index)
    return callee


@pytest.mark.parametrize(
    "callee, arguments, shape",
    [
        (A.IdentifierNode("float[2]"), 2, ("float", 2)),
        (A.IdentifierNode("float[N]"), 3, ("float", 3)),
        (_indexed("float", 2, 3), 2, ("float[3]", 2)),
        (_indexed("vec3", "N", 4), 2, ("vec3[4]", 2)),
        (_indexed("float[2]", 3), 3, ("float[3]", 2)),
        # A call through an indexed value, or a type index that is not a size, is not an array constructor.
        (_indexed("table", 0), 1, None),
        (_indexed("table", 0, 1), 1, None),
        (
            A.ArrayAccessNode(
                A.IdentifierNode("float"),
                A.BinaryOpNode(A.IdentifierNode("n"), "+", A.IdentifierNode("m")),
            ),
            1,
            None,
        ),
    ],
)
def test_array_constructor_shapes(callee, arguments, shape):
    call = A.FunctionCallNode(
        callee, [A.LiteralNode(1.0, A.PrimitiveType("float"))] * arguments
    )
    assert array_constructor_shape(call) == shape


_SPIRV_TOOLS = all(shutil.which(tool) for tool in ("spirv-as", "spirv-val"))


@pytest.mark.skipif(not _SPIRV_TOOLS, reason="spirv-as and spirv-val are not installed")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_spirv_output_validates(tmp_path, case):
    scope, body, _ = _CASES[case]
    (tmp_path / "program.spvasm").write_text(
        _translate(tmp_path, "vulkan", scope, body)
    )
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", "program.spvasm", "-o", "p.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "p.spv"],
    ):
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


_XCRUN = shutil.which("xcrun") is not None and (
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, check=False
    ).returncode
    == 0
)


@pytest.mark.skipif(not _XCRUN, reason="the Metal compiler is not installed")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_metal_output_compiles(tmp_path, case):
    scope, body, _ = _CASES[case]
    (tmp_path / "program.metal").write_text(_translate(tmp_path, "metal", scope, body))
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "program.metal", "-o", "p.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

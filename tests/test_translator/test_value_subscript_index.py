"""A declared variable's brackets are a subscript, even when the index is an assignment.

``t[j = k + 1]`` reads like named specialization arguments (``name = value``) to the CrossGL parser, which then
kept the whole access as the text of one identifier: GLSL, HLSL and Metal output repeated it verbatim and SPIR-V
refused an "unknown variable". A declared variable cannot be specialized, so its brackets now always hold an index
expression, unless the variable has parameters of its own or a call follows the brackets.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import (
    ArrayAccessNode,
    AssignmentNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import CrossGLFunctionBodyParseError, Parser

SHADER = """shader P {
    struct Pair { float arr[2]; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        float g[4];
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float t[4];
            Pair s;
            int k = 0;
            int j;
            t[0] = 1.0; t[1] = 2.0; t[2] = 3.0; t[3] = 4.0;
            g[0] = 5.0; g[1] = 6.0;
            s.arr[0] = 7.0; s.arr[1] = 8.0;
            float a = t[j = k + 1];
            float c = g[j = 1] + 1.0;
            float d = s.arr[j = 0];
            outp[j = 0] = a + c * 16.0 + d * 256.0;
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    if isinstance(node, (list, tuple)):
        children = node
    elif isinstance(node, dict):
        children = node.values()
    elif hasattr(node, "__dict__"):
        children = [value for key, value in vars(node).items() if key != "parent"]
    else:
        return
    for child in children:
        yield from _walk(child, seen)


def _declaration(ast, name):
    return next(
        node
        for node in _walk(ast)
        if isinstance(node, VariableNode) and node.name == name
    )


@pytest.mark.parametrize("name", ["a", "d"])
def test_assignment_index_parses_as_a_subscript(name):
    initial = _declaration(_parse(SHADER), name).initial_value
    assert isinstance(initial, ArrayAccessNode)
    assert isinstance(initial.index, AssignmentNode)
    array = initial.array
    if name == "d":
        assert isinstance(array, MemberAccessNode)
    else:
        assert isinstance(array, IdentifierNode) and array.name == "t"


def test_no_identifier_holds_subscript_text():
    names = [
        node.name for node in _walk(_parse(SHADER)) if isinstance(node, IdentifierNode)
    ]
    assert not [name for name in names if "[" in str(name)]


def test_comma_index_is_refused_rather_than_kept_as_text(tmp_path):
    path = tmp_path / "comma.cgl"
    path.write_text(
        "shader P { compute { void main() {"
        " float t[4]; int k = 0; float b = t[k, 2]; } } }"
    )
    with pytest.raises(CrossGLFunctionBodyParseError, match="Expected RBRACKET"):
        crosstl.translate(str(path), backend="opengl")


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan"])
def test_targets_index_with_the_assignment(tmp_path, backend):
    path = tmp_path / "subscript.cgl"
    path.write_text(SHADER)
    code = crosstl.translate(str(path), backend=backend)
    if backend == "vulkan":
        assert "Unknown variable" not in code
        if shutil.which("spirv-as") and shutil.which("spirv-val"):
            assembly = tmp_path / "subscript.spvasm"
            binary = tmp_path / "subscript.spv"
            assembly.write_text(code)
            subprocess.run(
                ["spirv-as", "--target-env", "vulkan1.1", str(assembly)]
                + ["-o", str(binary)],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
    else:
        assert "t[(j = " in code
        assert "g[(j = 1)]" in code


@pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="glslang")
def test_glslang_accepts_the_assignment_index(tmp_path):
    path = tmp_path / "subscript.cgl"
    path.write_text(SHADER)
    shader = tmp_path / "subscript.comp"
    shader.write_text(crosstl.translate(str(path), backend="opengl"))
    result = subprocess.run(
        ["glslangValidator", str(shader)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


def test_parameterized_values_keep_their_specialization():
    ast = _parse("""shader P {
        float Pick[n: Int] = 1.0;
        compute {
            void main() {
                float x = Pick[a = 2];
            }
        }
    }""")
    initial = _declaration(ast, "x").initial_value
    assert not isinstance(initial, ArrayAccessNode)

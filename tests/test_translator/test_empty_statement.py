"""CrossGL accepts an empty statement, a ``;`` on its own (issue #3292).

GLSL 4.60 section 6.1 and C++ [stmt.expr]/1, which HLSL, MSL, CUDA, HIP and Slang follow, make a lone ``;`` a
statement that does nothing. The CrossGL parser refused it, so every target refused a function containing one. It
now adds no node to the enclosing block, and a braceless ``if``, ``for`` or ``while`` body made of one is an empty
block.
"""

import shutil
import subprocess
import sys

import pytest

import crosstl
from crosstl.translator.ast import (
    BlockNode,
    BreakNode,
    ExpressionStatementNode,
    ForNode,
    IfNode,
    SwitchNode,
    WhileNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

EMPTY = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            int k = 0;
            ;
            ;;
            k = 1;
            if (iin[i] > 4) ;
            for (int j = 0; j < 3; j++) ;
            while (k > 1) ;
            switch (iin[i]) {
                case 0:
                    ;
                    k = 2;
                    break;
                default:
                    ;
            }
            iin[i] = k;
        }
    }
}
"""


def _main_body():
    shader = Parser(Lexer(EMPTY).get_tokens()).parse()
    (stage,) = shader.stages.values()
    return stage.entry_point.body


def _statements(body):
    return body.statements if isinstance(body, BlockNode) else body


def test_empty_statements_add_no_nodes():
    statements = _statements(_main_body())
    assert not any(isinstance(node, list) for node in statements)
    kinds = [type(node).__name__ for node in statements]
    # i, k, k = 1, if, for, while, switch, iin[i] = k: the three lone semicolons leave nothing behind.
    assert len(statements) == 8, kinds
    braceless = [
        node for node in statements if isinstance(node, (IfNode, ForNode, WhileNode))
    ]
    assert len(braceless) == 3
    for node in braceless:
        body = node.if_body if isinstance(node, IfNode) else node.body
        assert isinstance(body, BlockNode)
        assert body.statements == []


@pytest.mark.parametrize(
    "backend",
    ["opengl", "directx", "metal", "vulkan", "slang", "cuda", "hip", "cgl"],
)
def test_targets_translate_empty_statements(tmp_path, backend):
    path = tmp_path / "semi.cgl"
    path.write_text(EMPTY)
    code = crosstl.translate(str(path), backend=backend, format_output=False)
    assert "BlockNode" not in code
    assert all(line.strip() not in {"[]", "[];"} for line in code.splitlines())
    if backend != "vulkan":
        compact = " ".join(code.split())
        assert "k = 1;" in compact
        assert "k = 2;" in compact


def test_last_empty_label_leaves_the_switch():
    (switch,) = (
        node for node in _statements(_main_body()) if isinstance(node, SwitchNode)
    )
    first, last = switch.cases
    assert [type(node) for node in first.statements] == [
        ExpressionStatementNode,
        BreakNode,
    ]
    # MSL and GLSL ES 3.00 refuse a label followed by no statement; leaving the switch is what it already does.
    assert [type(node) for node in last.statements] == [BreakNode]


def _native(tmp_path, backend, suffix):
    path = tmp_path / "semi.cgl"
    path.write_text(EMPTY)
    output = tmp_path / f"semi.{suffix}"
    output.write_text(crosstl.translate(str(path), backend=backend))
    return output


def test_glslang_accepts_empty_statements(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    source = _native(tmp_path, "opengl", "comp")
    result = subprocess.run([glslang, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not followed by statements" not in result.stdout


def test_metal_accepts_empty_statements(tmp_path):
    if sys.platform != "darwin" or shutil.which("xcrun") is None:
        pytest.skip("the Metal compiler needs macOS")
    source = _native(tmp_path, "metal", "metal")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "semi.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr

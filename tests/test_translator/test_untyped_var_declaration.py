"""``var`` declarations without a written type take their initializer's type wherever a ``var`` may appear.

Before, ``var name = expr`` parsed only as a function-body statement: as a ``for`` initializer it failed to parse, and at
shader scope it was skipped, leaving later uses of the name undeclared.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.ast import ExpressionStatementNode, ForNode, VariableNode
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

LOOP_PROGRAM = textwrap.dedent("""
    shader main {
        compute {
            buffer float data[];
            void main() {
                for (var i = 0; i < 4; i++) {
                    data[i] = 1.0;
                }
                data[0] = 2.0;
            }
        }
    }
    """)

SHADER_SCOPE_PROGRAM = textwrap.dedent("""
    shader main {
        var scale = 2.0;
        compute {
            buffer float data[];
            void main() {
                data[0] = scale;
            }
        }
    }
    """)


def parse_strict(code):
    lexer = Lexer(code)
    lexer.tokenize()
    return Parser(lexer.tokens, strict_function_bodies=True).parse()


def test_untyped_var_declaration_preserves_function_body():
    code = """
shader main {
    compute {
        buffer float data[];
        void main() {
            var i = 1;
            data[0] = 2.0;
        }
    }
}
"""
    lexer = Lexer(code)
    lexer.tokenize()
    parser = Parser(lexer.tokens)
    ast = parser.parse()

    stage = list(ast.stages.values())[0]
    entry_point = stage.entry_point
    assert entry_point is not None, "Entry point function must exist"
    assert (
        len(entry_point.body.statements) == 2
    ), "Both var i = 1 and data[0] = 2.0 must be preserved"

    # Verify GLSL codegen
    glsl = GLSLCodeGen().generate(ast)
    assert "data[0] = 2.0;" in glsl, "data[0] assignment must be in GLSL output"
    assert "i = 1;" in glsl, "var i must be generated in GLSL output"

    # Verify Metal codegen
    metal = MetalCodeGen().generate(ast)
    assert "data[0] = 2.0;" in metal, "data[0] assignment must be in Metal output"
    assert "i = 1;" in metal, "var i must be generated in Metal output"


def test_untyped_var_initializes_a_for_loop():
    ast = parse_strict(LOOP_PROGRAM)
    statements = list(ast.stages.values())[0].entry_point.body.statements

    assert [type(statement) for statement in statements] == [
        ForNode,
        ExpressionStatementNode,
    ]
    init = statements[0].init
    assert isinstance(init, VariableNode)
    assert init.name == "i"
    assert init.var_type is None
    assert init.initial_value.value == 0
    assert len(statements[0].body.statements) == 1


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx"])
def test_untyped_for_loop_var_takes_its_initializers_type(tmp_path, backend):
    source = tmp_path / "loop.cgl"
    source.write_text(LOOP_PROGRAM)
    output = crosstl.translate(str(source), backend=backend)

    assert "for (int i = 0;" in output
    assert "data[0] = 2.0;" in output


def test_untyped_var_at_shader_scope_is_kept_as_a_global():
    ast = parse_strict(SHADER_SCOPE_PROGRAM)
    [scale] = [
        variable
        for variable in ast.global_variables
        if getattr(variable, "name", None) == "scale"
    ]

    assert scale.var_type is None
    assert scale.initial_value.value == 2.0
    assert "float scale = 2.0;" in GLSLCodeGen().generate(ast)

@pytest.mark.parametrize(
    ("initializer", "expected_type"),
    [("2.0", "f32"), ("2", "i32"), ("true", "bool")],
)
def test_untyped_shader_scope_var_uses_initializer_type_in_wgsl(
    tmp_path, initializer, expected_type
):
    source = tmp_path / "global.cgl"
    source.write_text(SHADER_SCOPE_PROGRAM.replace("2.0", initializer))

    output = crosstl.translate(str(source), backend="wgsl")

    assert f"var<private> scale: {expected_type} = {initializer};" in output
    assert "data[0] = scale;" in output


def test_untyped_shader_scope_var_without_inferable_initializer_fails_in_wgsl(
    tmp_path,
):
    source = tmp_path / "global.cgl"
    source.write_text(SHADER_SCOPE_PROGRAM.replace("2.0", "unknown_value"))

    with pytest.raises(ValueError, match="cannot infer the type of untyped global scale"):
        crosstl.translate(str(source), backend="wgsl")

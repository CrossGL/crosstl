from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


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
    assert len(entry_point.body.statements) == 2, "Both var i = 1 and data[0] = 2.0 must be preserved"

    # Verify GLSL codegen
    glsl = GLSLCodeGen().generate(ast)
    assert "data[0] = 2.0;" in glsl, "data[0] assignment must be in GLSL output"
    assert "i = 1;" in glsl, "var i must be generated in GLSL output"

    # Verify Metal codegen
    metal = MetalCodeGen().generate(ast)
    assert "data[0] = 2.0;" in metal, "data[0] assignment must be in Metal output"
    assert "i = 1;" in metal, "var i must be generated in Metal output"

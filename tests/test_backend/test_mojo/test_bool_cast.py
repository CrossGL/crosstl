"""Issue #3148: bool -> float/half casts compile on Mojo 1.2.0."""

from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def _compile(shader_src: str) -> str:
    ast = CrossGLParser(CrossGLLexer(shader_src).tokens).parse()
    return MojoCodeGen().generate(ast)


def test_bool_param_to_float_uses_scalar_cast():
    src = """
shader FB {
    float f(bool s) {
        return float(s);
    }
}
"""
    out = _compile(src)
    assert "Scalar[DType.bool](s).cast[DType.float32]()" in out, out
    assert "Float32(s)" not in out


def test_comparison_result_to_float_uses_scalar_cast():
    src = """
shader FB {
    float g(float x) {
        return float(x < 1.0);
    }
}
"""
    out = _compile(src)
    assert "Scalar[DType.bool]" in out, out
    assert "Float32((x < 1.0))" not in out


def test_bool_param_to_half_uses_scalar_cast():
    src = """
shader FB {
    half h(bool s) {
        return half(s);
    }
}
"""
    out = _compile(src)
    assert "Scalar[DType.bool](s).cast[DType.float16]()" in out, out

"""Alias resolution must give the same answers when the plain-name fast paths are taken."""

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser


def convert(code: str) -> str:
    ast = MetalParser(MetalLexer(code).tokenize()).parse()
    return MetalToCrossGLConverter().generate(ast)


def test_plain_type_names_are_returned_unchanged():
    converter = MetalToCrossGLConverter()

    assert converter.materialize_alias_template_type("  float4 ") == "float4"
    assert converter.materialize_alias_template_type("device float*") == (
        "device float*"
    )
    assert converter.materialize_alias_template_type("") == ""
    assert converter.materialize_alias_template_type(None) == ""
    assert converter.scalar_alias_binding("float") is None
    assert converter.scalar_alias_binding("metal::float") is None


def test_namespace_qualified_scalar_alias_still_resolves():
    crossgl = convert("""
        namespace precision {
        typedef float real;
        }

        kernel void scale(device precision::real* values [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
            precision::real factor = precision::real(2.0f);
            values[gid] = values[gid] * factor;
        }
        """)

    assert "float factor = float(2.0f);" in crossgl
    assert "precision::real" not in crossgl


def test_alias_template_instance_still_resolves():
    crossgl = convert("""
        template <typename T>
        using Pair = vec<T, 2>;

        kernel void fill(device float2* values [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
            Pair<float> value = values[gid];
            values[gid] = value * 2.0f;
        }
        """)

    assert "Pair" not in crossgl
    assert "vec2 value = " in crossgl

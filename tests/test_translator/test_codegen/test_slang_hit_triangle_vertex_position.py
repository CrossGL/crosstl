"""Slang output of GL_EXT_ray_tracing_position_fetch's ``gl_HitTriangleVertexPositionsEXT``.

Slang declares no such array: a hit shader reads one vertex position at a time with
``HitTriangleVertexPosition(index)``, so slangc rejected the name the generator used to print.
"""

import pytest

from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def closest_hit(body):
    return f"""
shader Probe {{
    struct Payload {{
        vec4 color;
    }}
    ray_closest_hit {{
        void main(Payload payload @ payload, BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {{
            {body}
        }}
    }}
}}
"""


def generate(source):
    return SlangCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def test_an_element_is_read_through_hit_triangle_vertex_position():
    code = generate(
        closest_hit(
            "uint k = 2u;\n"
            "vec3 v = gl_HitTriangleVertexPositionsEXT[1];\n"
            "payload.color = vec4(v + gl_HitTriangleVertexPositionsEXT[k], 1.0);"
        )
    )
    assert "float3 v = HitTriangleVertexPosition(1);" in code
    assert "HitTriangleVertexPosition(k)" in code
    assert "gl_HitTriangleVertexPositionsEXT" not in code


def test_the_whole_array_is_refused():
    with pytest.raises(SlangLoweringRefused, match="HitTriangleVertexPosition"):
        generate(
            closest_hit(
                "vec3 w[3] = gl_HitTriangleVertexPositionsEXT;\n"
                "payload.color = vec4(w[0], 1.0);"
            )
        )

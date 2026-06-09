import pytest

import crosstl.translator
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

GEOMETRY_GLSL = """
#version 450 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vColor[];
out vec3 gColor;

void main() {
    for (int i = 0; i < 3; i++) {
        gColor = vColor[i];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""


GEOMETRY_ADJACENCY_LAYOUT_GLSL = """
#version 450 core
layout(lines_adjacency, invocations = 2) in;
layout(triangle_strip, max_vertices = 6) out;

void main() {
    EmitVertex();
    EndPrimitive();
}
"""


GEOMETRY_GL_PERVERTEX_REDECLARATION_GLSL = """
#version 450 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in gl_PerVertex {
    vec4 gl_Position;
} gl_in[];

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def generate_crossgl(code: str, shader_type: str):
    ast = parse_glsl(code, shader_type)
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(ast)


def parse_crossgl(code: str):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def regenerate_glsl(code: str, shader_type: str):
    return GLSLCodeGen().generate(
        crosstl.translator.parse(generate_crossgl(code, shader_type))
    )


def test_parse_geometry_shader():
    ast = parse_glsl(GEOMETRY_GLSL, "geometry")
    assert ast is not None
    assert ast.functions


def test_codegen_geometry_roundtrip():
    output = generate_crossgl(GEOMETRY_GLSL, "geometry")
    assert "geometry" in output.lower()
    assert "EmitVertex" in output
    assert "in vec3 vColor[];" in output
    assert "out vec3 gColor;" in output
    assert "GeometryInput" not in output
    assert "GeometryOutput" not in output
    shader_ast = parse_crossgl(output)
    assert ShaderStage.GEOMETRY in shader_ast.stages

    glsl = regenerate_glsl(GEOMETRY_GLSL, "geometry")
    assert "in vec3 vColor[];" in glsl
    assert "out vec3 gColor;" in glsl
    assert "GeometryInput" not in glsl
    assert "GeometryOutput" not in glsl
    assert "input." not in glsl
    assert "output." not in glsl


def test_codegen_geometry_layout_roundtrip_preserves_adjacency_and_invocations():
    crossgl = generate_crossgl(GEOMETRY_ADJACENCY_LAYOUT_GLSL, "geometry")

    assert "layout(lines_adjacency, invocations = 2) in;" in crossgl
    assert "layout(triangle_strip, max_vertices = 6) out;" in crossgl
    assert "// layout(" not in crossgl

    glsl = regenerate_glsl(GEOMETRY_ADJACENCY_LAYOUT_GLSL, "geometry")

    assert "layout(lines_adjacency, invocations = 2) in;" in glsl
    assert "layout(triangle_strip, max_vertices = 6) out;" in glsl
    assert "return output;" not in glsl


def test_codegen_geometry_repeated_stream_layouts_from_glslang_roundtrip():
    # Reduced from KhronosGroup/glslang Test/glsl.nvgpushader5.geom.
    code = """
    #version 150
    #extension GL_NV_gpu_shader5 : enable

    sample in vec4 colorSampIn[3];
    sample out vec4 colorSampOut;

    layout(triangles, invocations = 6) in;
    layout(points, stream = 0) out;
    layout(stream = 1) out;

    void main() {
        EmitStreamVertex(1);
        EndStreamPrimitive(0);
    }
    """

    crossgl = generate_crossgl(code, "geometry")

    assert "sample in vec4 colorSampIn[3];" in crossgl
    assert "sample out vec4 colorSampOut;" in crossgl
    assert "layout(points, stream = 0) out;" in crossgl
    assert "layout(stream = 1) out;" in crossgl
    parse_crossgl(crossgl)

    glsl = regenerate_glsl(code, "geometry")

    assert "layout(points, stream = 0) out;" in glsl
    assert "layout(stream = 1) out;" in glsl


def test_codegen_geometry_builtin_gl_pervertex_redeclarations_roundtrip():
    # Reduced from Khronos GLSL 4.60.8 section 7.5, which documents
    # redeclaring the built-in gl_PerVertex block.
    crossgl = generate_crossgl(GEOMETRY_GL_PERVERTEX_REDECLARATION_GLSL, "geometry")

    assert crossgl.count("struct gl_PerVertex") == 1

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert glsl.count("gl_PerVertex") == 1
    assert "gl_Position = gl_in[i].gl_Position;" in glsl
    assert "EmitVertex();" in glsl


if __name__ == "__main__":
    pytest.main()

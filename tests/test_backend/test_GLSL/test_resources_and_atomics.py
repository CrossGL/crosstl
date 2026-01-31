import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


RESOURCE_GLSL = """
#version 450 core
layout(binding = 0, rgba8) uniform image2D outputImage;
layout(binding = 1) uniform sampler2D tex;
layout(binding = 2) uniform isampler2D itex;
layout(binding = 3) uniform usampler2D utex;

layout(std430, binding = 4) buffer DataBlock {
    vec4 values[];
} dataBlock;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 fragColor;

uniform atomic_uint counter;

void main() {
    vec4 c = imageLoad(outputImage, ivec2(0, 0));
    vec4 t = texture(tex, vUV);
    imageStore(outputImage, ivec2(0, 0), c + t);
    uint prev = atomicAdd(counter, 1);
    memoryBarrier();
    barrier();
    fragColor = t + vec4(float(prev));
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


def test_parse_resources_and_atomics():
    ast = parse_glsl(RESOURCE_GLSL, "fragment")
    assert ast is not None


def test_parse_sampler_image_variants():
    code = """
    #version 450 core
    uniform sampler1D s1d;
    uniform sampler2DRect srect;
    uniform sampler2DArrayShadow s2da;
    uniform samplerCubeArray sca;
    uniform isampler2D is2d;
    uniform usampler2D us2d;
    layout(binding = 1) uniform image1D img1d;
    layout(binding = 2) uniform image2DArray img2da;
    layout(binding = 3) uniform imageBuffer imgBuf;
    void main() { }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_parse_image_atomics_and_counters():
    code = """
    #version 450 core
    layout(binding = 0, r32ui) uniform uimage2D img;
    layout(binding = 1) uniform atomic_uint counter;
    void main() {
        uint old = imageAtomicAdd(img, ivec2(0), 1u);
        uint next = atomicCounterIncrement(counter);
        atomicCounterDecrement(counter);
        memoryBarrier();
        barrier();
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_codegen_resources_roundtrip():
    output = generate_crossgl(RESOURCE_GLSL, "fragment")
    assert "imageLoad" in output
    assert "imageStore" in output
    assert "atomicAdd" in output
    shader_ast = parse_crossgl(output)
    assert shader_ast is not None


if __name__ == "__main__":
    pytest.main()

import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def test_parse_texture_sampling_variants():
    code = """
    #version 450 core
    layout(location = 0) in vec2 vUV;
    layout(location = 0) out vec4 fragColor;
    uniform sampler2D tex;

    void main() {
        vec4 c0 = texture(tex, vUV);
        vec4 c1 = textureLod(tex, vUV, 0.0);
        vec4 c2 = textureGrad(tex, vUV, vec2(1.0), vec2(1.0));
        vec4 c3 = texelFetch(tex, ivec2(0), 0);
        vec4 c4 = textureGather(tex, vUV);
        fragColor = c0 + c1 + c2 + c3 + c4;
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_parse_texture_queries():
    code = """
    #version 450 core
    uniform sampler2D tex;
    void main() {
        ivec2 size = textureSize(tex, 0);
        int levels = textureQueryLevels(tex);
        vec2 lod = textureQueryLod(tex, vec2(0.5));
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


if __name__ == "__main__":
    pytest.main()

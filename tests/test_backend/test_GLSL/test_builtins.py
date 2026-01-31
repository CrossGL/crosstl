import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def test_vertex_builtins_parse():
    code = """
    #version 450 core
    layout(location = 0) in vec3 position;
    void main() {
        int id = gl_VertexID + gl_InstanceID;
        gl_Position = vec4(position, 1.0);
    }
    """
    ast = parse_glsl(code, "vertex")
    assert ast is not None


def test_fragment_builtins_parse():
    code = """
    #version 450 core
    layout(location = 0) out vec4 fragColor;
    void main() {
        vec2 p = gl_FragCoord.xy;
        if (gl_FrontFacing) {
            fragColor = vec4(p, 0.0, 1.0);
        } else {
            fragColor = vec4(0.0);
        }
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_compute_builtins_parse():
    code = """
    #version 430 core
    layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    void main() {
        uvec3 gid = gl_WorkGroupID;
        uvec3 lid = gl_LocalInvocationID;
        uvec3 ginv = gl_GlobalInvocationID;
    }
    """
    ast = parse_glsl(code, "compute")
    assert ast is not None


if __name__ == "__main__":
    pytest.main()

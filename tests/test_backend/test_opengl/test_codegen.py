import pytest

from crosstl.backend.OpenGL.OpenglLexer import GLSLLexer
from crosstl.backend.OpenGL.OpenglParser import GLSLParser
from crosstl.backend.OpenGL.OpenglCrossGLCodeGen import GLSLToCrossGLConverter


# Helper functions for tokenizing and parsing GLSL code
def tokenize(input_str):
    """Helper function to tokenize GLSL input string"""
    lexer = GLSLLexer(input_str)
    return lexer.tokenize()


def parse(tokens, shader_type="vertex"):
    """Helper function to parse GLSL tokens"""
    parser = GLSLParser(tokens, shader_type)
    return parser.parse()


def generate(ast, shader_type="vertex"):
    """Helper function to generate CrossGL code from GLSL AST"""
    generator = GLSLToCrossGLConverter(shader_type=shader_type)
    return generator.generate(ast)


def process_glsl(input_str, shader_type="vertex"):
    """Process complete GLSL code to CrossGL"""
    tokens = tokenize(input_str)
    ast = parse(tokens, shader_type)
    return generate(ast, shader_type)


# Tests for the OpenGL to CrossGL converter
def test_empty_shader():
    result = process_glsl("")
    assert "shader main" in result
    assert "vertex" in result


def test_basic_vertex_shader():
    glsl_code = """
    void main() {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    }
    """
    result = process_glsl(glsl_code, "vertex")
    assert "shader main" in result
    assert "vertex" in result
    assert "vec4" in result


def test_variable_declarations():
    glsl_code = """
    void main() {
        float x = 1.0;
        vec3 pos = vec3(x, 2.0, 3.0);
        gl_Position = vec4(pos, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    # Check that key variables and values are in the result
    assert "x" in result and "1.0" in result
    assert "pos" in result and "vec3" in result
    assert "gl_Position" in result and "vec4" in result


def test_arithmetic_expressions():
    glsl_code = """
    void main() {
        float x = 1.0 + 2.0 * 3.0;
        float y = x / 2.0 - 1.0;
        gl_Position = vec4(x, y, 0.0, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    # Check that values and variables are present
    assert "1.0" in result and "2.0" in result and "3.0" in result
    assert "x" in result and "y" in result
    assert "gl_Position" in result and "vec4" in result


def test_function_calls():
    glsl_code = """
    float compute(float a, float b) {
        return a * b + 1.0;
    }
    
    void main() {
        float result = compute(2.0, 3.0);
        gl_Position = vec4(result, 0.0, 0.0, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    assert "float compute" in result
    assert "return" in result
    assert "compute" in result and "2.0" in result and "3.0" in result


def test_if_statements():
    glsl_code = """
    void main() {
        float x = 1.0;
        if (x > 0.5) {
            x = 2.0;
        } else {
            x = 0.5;
        }
        gl_Position = vec4(x, 0.0, 0.0, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    assert "if" in result
    assert "else" in result
    assert "x" in result and "0.5" in result


def test_for_loop():
    glsl_code = """
    void main() {
        float sum = 0.0;
        for (int i = 0; i < 10; i++) {
            sum += float(i);
        }
        gl_Position = vec4(sum, 0.0, 0.0, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    assert "for" in result
    assert "i = 0" in result
    assert "i < 10" in result or "i<10" in result
    assert "sum" in result and "float" in result
    assert "gl_Position" in result and "vec4" in result


def test_vertex_shader_with_attributes():
    glsl_code = """
    attribute vec3 aPosition;
    attribute vec2 aTexCoord;

    varying vec2 vTexCoord;

    uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;

    void main() {
        vTexCoord = aTexCoord;
        gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPosition, 1.0);
    }
    """
    result = process_glsl(glsl_code, "vertex")
    assert "VertexInput" in result
    assert "VertexOutput" in result
    assert "cbuffer Uniforms" in result
    assert "uModelViewMatrix" in result and "uProjectionMatrix" in result
    assert "mat4" in result
    assert "vTexCoord" in result and "aTexCoord" in result
    assert "gl_Position" in result and "aPosition" in result


def test_fragment_shader():
    glsl_code = """
    varying vec2 vTexCoord;
    
    uniform sampler2D uTexture;
    
    void main() {
        gl_FragColor = texture2D(uTexture, vTexCoord);
    }
    """
    result = process_glsl(glsl_code, "fragment")
    assert "fragment" in result
    assert "FragmentInput" in result
    assert "sample" in result


def test_vector_constructor():
    glsl_code = """
    void main() {
        vec2 v2 = vec2(1.0, 2.0);
        vec3 v3 = vec3(v2, 3.0);
        vec4 v4 = vec4(v3, 1.0);
        gl_Position = v4;
    }
    """
    result = process_glsl(glsl_code)
    assert "vec2" in result and "1.0" in result and "2.0" in result
    assert "vec3" in result and "v2" in result and "3.0" in result
    assert "vec4" in result and "v3" in result and "1.0" in result


def test_builtin_functions():
    glsl_code = """
    void main() {
        vec3 a = vec3(1.0, 2.0, 3.0);
        vec3 b = vec3(4.0, 5.0, 6.0);
        float d = dot(normalize(a), b);
        vec3 r = reflect(a, normalize(b));
        gl_Position = vec4(r, d);
    }
    """
    result = process_glsl(glsl_code)
    assert "dot" in result
    assert "normalize" in result
    assert "reflect" in result


def test_struct_definition():
    glsl_code = """
    struct Light {
        vec3 position;
        vec3 color;
        float intensity;
    };
    
    uniform Light uLight;
    
    void main() {
        vec3 lightPos = uLight.position;
        gl_Position = vec4(lightPos, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    assert "struct Light" in result
    assert "vec3" in result and "position" in result
    assert "vec3" in result and "color" in result
    assert "float" in result and "intensity" in result
    assert "Light" in result and "uLight" in result


def test_array_access():
    glsl_code = """
    uniform float uValues[4];

    void main() {
        float sum = 0.0;
        for (int i = 0; i < 4; i++) {
            sum += uValues[i];
        }
        gl_Position = vec4(sum, 0.0, 0.0, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    assert "uValues" in result
    assert "sum" in result
    assert "for" in result
    assert "i" in result
    assert "gl_Position" in result and "vec4" in result


def test_ternary_operator():
    glsl_code = """
    void main() {
        float x = 0.5;
        float y = x > 0.5 ? 1.0 : 0.0;
        gl_Position = vec4(y, 0.0, 0.0, 1.0);
    }
    """
    result = process_glsl(glsl_code)
    assert "?" in result
    assert ":" in result
    assert "x" in result and "0.5" in result


def test_member_access():
    glsl_code = """
    varying vec4 vColor;
    
    void main() {
        float r = vColor.r;
        float g = vColor.g;
        float b = vColor.b;
        float a = vColor.a;
        gl_FragColor = vec4(r, g, b, a);
    }
    """
    result = process_glsl(glsl_code, "fragment")
    assert "vColor.r" in result.replace(" ", "")
    assert "vColor.g" in result.replace(" ", "")
    assert "vColor.b" in result.replace(" ", "")
    assert "vColor.a" in result.replace(" ", "")


if __name__ == "__main__":
    pytest.main()

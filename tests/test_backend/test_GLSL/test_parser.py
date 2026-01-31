import textwrap
from typing import List

import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def tokenize_code(code: str) -> List:
    """Helper function to tokenize GLSL code."""
    lexer = GLSLLexer(code)
    return lexer.tokenize()


def parse_code(code: str, shader_type: str = "vertex"):
    tokens = tokenize_code(code)
    parser = GLSLParser(tokens, shader_type)
    return parser.parse()


def parse_ok(code: str, shader_type: str = "vertex"):
    ast = parse_code(code, shader_type)
    assert ast is not None
    return ast


def parse_fails(code: str, shader_type: str = "vertex"):
    with pytest.raises(SyntaxError):
        parse_code(code, shader_type)


def test_parse_vertex_shader_with_layout_and_io():
    code = textwrap.dedent(
        """
        #version 450 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 uv;
        layout(location = 0) out vec2 vUV;
        uniform mat4 uMVP;

        void main() {
            vUV = uv;
            gl_Position = uMVP * vec4(position, 1.0);
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_fragment_shader_with_discard():
    code = textwrap.dedent(
        """
        #version 450 core
        layout(location = 0) in vec2 vUV;
        layout(location = 0) out vec4 fragColor;
        uniform sampler2D uTexture;

        void main() {
            vec4 color = texture(uTexture, vUV);
            if (color.a < 0.1) {
                discard;
            }
            fragColor = color;
        }
        """
    )
    parse_ok(code, "fragment")


def test_parse_structs_and_arrays():
    code = textwrap.dedent(
        """
        #version 450 core
        struct Light {
            vec3 position;
            vec3 color;
            float intensity;
        };

        uniform Light lights[4];
        uniform float weights[4];
        layout(location = 0) out vec3 vColor;

        void main() {
            vec3 color = lights[0].color * weights[1];
            vColor = color;
            gl_Position = vec4(1.0);
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_control_flow_constructs():
    code = textwrap.dedent(
        """
        #version 450 core
        layout(location = 0) in vec3 position;

        void main() {
            int i = 0;
            for (int j = 0; j < 4; j++) {
                if (j == 2) {
                    continue;
                }
                i += j;
            }

            while (i < 10) {
                i++;
            }

            do {
                i--;
            } while (i > 0);

            switch (i) {
                case 0:
                    i = 1;
                    break;
                default:
                    break;
            }

            if (i > 2) {
                i = i * 2;
            } else {
                i = 1;
            }

            gl_Position = vec4(position, 1.0);
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_ternary_and_bitwise_expressions():
    code = textwrap.dedent(
        """
        #version 450 core
        void main() {
            int a = 3;
            int b = 5;
            int c = (a > b) ? a : b;
            int d = (a & b) | (a ^ b);
            int e = ~a;
            bool f = (a != b) && (a <= b || b >= a);
            gl_Position = vec4(float(c + d + e));
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_matrix_vector_operations():
    code = textwrap.dedent(
        """
        #version 450 core
        void main() {
            mat4 m = mat4(1.0);
            vec4 v = vec4(1.0, 2.0, 3.0, 1.0);
            vec4 r = m * v;
            vec3 n = normalize(v.xyz);
            gl_Position = r + vec4(n, 0.0);
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_precision_qualifiers():
    code = textwrap.dedent(
        """
        #version 300 es
        precision highp float;
        precision mediump sampler2D;
        in vec2 vUV;
        out vec4 fragColor;
        uniform sampler2D uTexture;

        void main() {
            mediump vec4 color = texture(uTexture, vUV);
            fragColor = color;
        }
        """
    )
    parse_ok(code, "fragment")


def test_parse_preprocessor_directives():
    code = textwrap.dedent(
        """
        #version 450 core
        #define USE_LIGHTING 1
        #if USE_LIGHTING
        #define FACTOR 0.5
        #endif

        void main() {
            float value = FACTOR;
            gl_Position = vec4(value);
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_function_overloads():
    code = textwrap.dedent(
        """
        #version 450 core
        float compute(vec3 n, vec3 l) {
            return max(dot(n, l), 0.0);
        }

        float compute(vec3 n, vec3 l, float intensity) {
            return compute(n, l) * intensity;
        }

        void main() {
            float v = compute(vec3(0.0), vec3(1.0), 0.5);
            gl_Position = vec4(v);
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_interface_blocks_and_uniform_block():
    code = textwrap.dedent(
        """
        #version 450 core
        layout(std140, binding = 0) uniform Globals {
            mat4 mvp;
            vec4 baseColor;
        };

        in VertexIn {
            vec3 position;
            vec2 uv;
        } vin;

        out VertexOut {
            vec4 color;
        } vout;

        void main() {
            vout.color = vec4(vin.position, 1.0);
            gl_Position = mvp * vec4(vin.position, 1.0);
        }
        """
    )
    parse_ok(code, "vertex")


def test_parse_compute_layout_qualifier():
    code = textwrap.dedent(
        """
        #version 430 core
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        void main() {
            // No-op
        }
        """
    )
    parse_ok(code, "compute")


def test_parse_subroutine_declaration():
    code = textwrap.dedent(
        """
        #version 450 core
        subroutine vec4 ShadeFunc(vec3 n);
        subroutine uniform ShadeFunc shade;
        void main() { }
        """
    )
    parse_ok(code, "fragment")


def test_parse_swizzle_and_constructors():
    code = textwrap.dedent(
        """
        #version 450 core
        void main() {
            vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
            vec2 xy = v.xy;
            vec3 rgb = v.rgb;
            mat3 m = mat3(1.0);
            float arr[] = float[](1.0, 2.0, 3.0);
        }
        """
    )
    parse_ok(code, "vertex")


@pytest.mark.parametrize(
    "code",
    [
        "void main() { int a = 0 }",  # Missing semicolon
        "void main() { if (true) { gl_Position = vec4(1.0); }",  # Missing brace
        "void main() { else { gl_Position = vec4(1.0); } }",  # Else without if
        "void main() { case 0: break; }",  # Case outside switch
        "void main() { switch (1) { case 0 return; } }",  # Missing colon
        "layout(location =) in vec3 position; void main() { gl_Position = vec4(position, 1.0); }",  # Bad layout
        "void main( { gl_Position = vec4(1.0); }",  # Unterminated signature
    ],
)
def test_parse_invalid_syntax_cases(code):
    parse_fails(code, "vertex")


if __name__ == "__main__":
    pytest.main()

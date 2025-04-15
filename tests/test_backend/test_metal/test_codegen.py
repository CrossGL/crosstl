import pytest
from typing import List
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = MetalLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = MetalParser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = MetalToCrossGLConverter()
    return codegen.generate(ast_node)


def test_struct():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex_INPUT {
        float3 position [[attribute(0)]];
    };

    struct Vertex_OUTPUT {
        float4 position [[position]];
        float2 vUV;
    };

    vertex Vertex_OUTPUT vertex_main(Vertex_INPUT input [[stage_in]]) {
        Vertex_OUTPUT output;
        output.position = float4(input.position, 1.0);
        return output;
    }

    struct Fragment_INPUT {
        float2 vUV [[stage_in]];
    };

    struct Fragment_OUTPUT {
        float4 fragColor [[color(0)]];
    };

    fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
        Fragment_OUTPUT output;
        output.fragColor = float4(1.0, 0.0, 0.0, 1.0);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex_INPUT {
        float3 position [[attribute(0)]];
    };

    struct Vertex_OUTPUT {
        float4 position [[position]];
        float2 vUV;
    };

    vertex Vertex_OUTPUT vertex_main(Vertex_INPUT input [[stage_in]]) {
        Vertex_OUTPUT output;
        output.position = float4(input.position, 1.0);
        if (input.position.x == input.position.y) {
            output.vUV = float2(0.0, 0.0);
        }
        return output;
    }

    struct Fragment_INPUT {
        float2 vUV [[stage_in]];
    };

    struct Fragment_OUTPUT {
        float4 fragColor [[color(0)]];
    };

    fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
        Fragment_OUTPUT output;
        output.fragColor = float4(1.0, 0.0, 0.0, 1.0);
        if (input.vUV.x == input.vUV.y) {
            output.fragColor = float4(0.0, 1.0, 0.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_for():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex_INPUT {
        float3 position [[attribute(0)]];
    };

    struct Vertex_OUTPUT {
        float4 position [[position]];
        float2 vUV;
    };

    vertex Vertex_OUTPUT vertex_main(Vertex_INPUT input [[stage_in]]) {
        Vertex_OUTPUT output;
        output.position = float4(input.position, 1.0);
        for (int i = 0; i < 10; i=i+1) {
            output.vUV = float2(0.0, 0.0);
        }
        return output;
    }

    struct Fragment_INPUT {
        float2 vUV [[stage_in]];
    };

    struct Fragment_OUTPUT {
        float4 fragColor [[color(0)]];
    };

    fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
        Fragment_OUTPUT output;
        output.fragColor = float4(1.0, 0.0, 0.0, 1.0);
        for (int i = 0; i < 10; i=i+1) {
            output.fragColor = float4(0.0, 1.0, 0.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex_INPUT {
        float3 position [[attribute(0)]];
    };

    struct Vertex_OUTPUT {
        float4 position [[position]];
        float2 vUV;
    };

    vertex Vertex_OUTPUT vertex_main(Vertex_INPUT input [[stage_in]]) {
        Vertex_OUTPUT output;
        output.position = float4(input.position, 1.0);
        if (input.position.x == input.position.y) {
            output.vUV = float2(0.0, 0.0);
        } else {
            output.vUV = float2(1.0, 1.0);
        }
        return output;
    }

    struct Fragment_INPUT {
        float2 vUV [[stage_in]];
    };

    struct Fragment_OUTPUT {
        float4 fragColor [[color(0)]];
    };

    fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
        Fragment_OUTPUT output;
        output.fragColor = float4(1.0, 0.0, 0.0, 1.0);
        if (input.vUV.x == input.vUV.y) {
            output.fragColor = float4(0.0, 1.0, 0.0, 1.0);
        } else {
            output.fragColor = float4(1.0, 0.0, 0.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_function_call():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    float perlinNoise(float2 p) {
        return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    }

    struct Vertex_INPUT {
        float3 position [[attribute(0)]];
    };

    struct Vertex_OUTPUT {
        float4 position [[position]];
        float2 vUV;
    };

    vertex Vertex_OUTPUT vertex_main(Vertex_INPUT input [[stage_in]]) {
        Vertex_OUTPUT output;
        output.position = float4(input.position, 1.0);
        output.vUV = perlinNoise(float2(0.0, 0.0));
        return output;
    }

    struct Fragment_INPUT {
        float2 vUV [[stage_in]];
    };

    struct Fragment_OUTPUT {
        float4 fragColor [[color(0)]];
    };

    fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
        Fragment_OUTPUT output;
        output.fragColor = float4(1.0, 0.0, 0.0, 1.0);
        float noise = perlinNoise(input.vUV);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_if():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex_INPUT {
        float3 position [[attribute(0)]];
    };

    struct Vertex_OUTPUT {
        float4 position [[position]];
        float2 vUV;
    };

    vertex Vertex_OUTPUT vertex_main(Vertex_INPUT input [[stage_in]]) {
        Vertex_OUTPUT output;
        output.position = float4(input.position, 1.0);
        if (input.position.x == input.position.y) {
            output.vUV = float2(0.0, 0.0);}
        
        else if (input.position.x > input.position.y) {
            output.vUV = float2(1.0, 1.0);
        } 
        else if (input.position.x < input.position.y) {
            output.vUV = float2(-1.0, -1.0);
        }
        else {
            output.vUV = float2(0.0, 0.0);
        }
        return output;
    }

    struct Fragment_INPUT {
        float2 vUV [[stage_in]];
    };

    struct Fragment_OUTPUT {
        float4 fragColor [[color(0)]];
    };

    fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
        Fragment_OUTPUT output;
        if (input.vUV.x == input.vUV.y) {
            output.fragColor = float4(0.0, 1.0, 0.0, 1.0);
        } else if (input.vUV.x > input.vUV.y) {
            output.fragColor = float4(1.0, 0.0, 0.0, 1.0);
        } else {
            output.fragColor = float4(0.0, 0.0, 1.0, 1.0);
        }
        return output;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    print(generated_code)


def test_bitwise_not_codegen():
    code = """
    void main() {
        int a = 5;
        int b = ~a;  // Bitwise NOT
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator code generation not implemented")


def test_const_codegen():
    code = """
    void main() {
        const float PI = 3.14159;
        const int VERSION = 100;
        float radius = 5.0;
        float area = PI * radius * radius;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert (
            "const float PI" in generated_code
        ), "Const declaration not properly generated"
    except SyntaxError as e:
        pytest.fail(f"Const variable code generation not implemented: {e}")


def test_texture_sampling():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    
    struct VertexOut {
        float4 position [[position]];
        float2 texCoord;
    };
    
    fragment float4 fragmentShader(VertexOut in [[stage_in]],
                                   texture2d<float> diffuseTexture [[texture(0)]],
                                   sampler textureSampler [[sampler(0)]]) {
        // Basic texture sampling
        float4 color1 = diffuseTexture.sample(textureSampler, in.texCoord);
        
        // Texture sampling with LOD
        float4 color2 = diffuseTexture.sample(textureSampler, in.texCoord, 2.0);
        
        return color1 * 0.5 + color2 * 0.5;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError as e:
        pytest.fail(f"Texture sampling code generation not implemented: {e}")


if __name__ == "__main__":
    pytest.main()

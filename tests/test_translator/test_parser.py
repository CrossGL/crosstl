from crosstl.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.translator.parser import Parser
from crosstl.translator.ast import (
    ArrayType,
    ForInNode,
    FunctionCallNode,
    LiteralNode,
    LoopNode,
    MatrixType,
    PreprocessorNode,
    PrimitiveType,
    RangeNode,
    ShaderNode,
    ShaderStage,
    VariableNode,
    VectorType,
    WaveOpNode,
    RayTracingOpNode,
    MeshOpNode,
    RayQueryOpNode,
)


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.tokens


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def test_struct_tokenization():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            }

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else {
                bloom = 0.0;
            }

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}

"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("if statement parsing not implemented.")


def test_for_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            for (int i = 0; i < 10; i++) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            }
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }
}

"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("for parsing not implemented.")


def test_loop_statement_parses_to_loop_node():
    code = """
    shader main {
        int helper(int limit) {
            int i = 0;
            loop {
                i = i + 1;
                if (i >= limit) {
                    break;
                }
            }
            return i;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    helper = ast.functions[0]

    assert any(isinstance(stmt, LoopNode) for stmt in helper.body.statements)


def test_for_in_statement_parses_to_for_in_node():
    code = """
    shader main {
        int helper(int limit) {
            int total = 0;
            for i in limit {
                total = total + i;
            }
            return total;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    helper = ast.functions[0]

    assert any(isinstance(stmt, ForInNode) for stmt in helper.body.statements)


def test_for_in_range_statement_parses_to_range_node():
    code = """
    shader main {
        int helper(int limit) {
            int total = 0;
            for i in 2..5 {
                total = total + i;
            }
            for j in 1..=limit {
                total = total + j;
            }
            return total;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    helper = ast.functions[0]
    ranges = [
        stmt.iterable for stmt in helper.body.statements if isinstance(stmt, ForInNode)
    ]

    assert len(ranges) == 2
    assert isinstance(ranges[0], RangeNode)
    assert ranges[0].start.value == 2
    assert ranges[0].end.value == 5
    assert not ranges[0].inclusive
    assert isinstance(ranges[1], RangeNode)
    assert ranges[1].start.value == 1
    assert ranges[1].end.name == "limit"
    assert ranges[1].inclusive


def test_range_expression_preserves_assignment_rhs():
    code = """
    shader main {
        int helper() {
            int total = 0;
            total = 2..5;
            return total;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    helper = ast.functions[0]
    assignment = helper.body.statements[1].expression

    assert isinstance(assignment.value, RangeNode)
    assert assignment.value.start.value == 2
    assert assignment.value.end.value == 5
    assert not isinstance(assignment.target, RangeNode)


def test_preprocessor_and_precision_parsing():
    code = """
    #version 450 core
    precision highp float;
    shader main {
        vertex {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assert isinstance(ast, ShaderNode)
    directives = [
        pp
        for pp in getattr(ast, "preprocessors", [])
        if isinstance(pp, PreprocessorNode)
    ]
    names = {pp.directive for pp in directives}
    assert "version" in names
    assert "precision" in names


def test_else_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else if (input.texCoord.x < 0.5) {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                output.color = vec4(0.5, 0.5, 0.5, 1.0);

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }
}

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else if (bloom < 0.5) {
                bloom = 0.0;
            } else {
                bloom = 0.5;
            }
            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}

"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("else if statement parsing not implemented.")


def test_function_call():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vec4 addColor(vec4 color1, vec4 color2) {
        return color1 + color2;
    }

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("function call parsing not implemented.")


def test_assign_shift_right():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            uint a = 8;
            a >>= 1;
            uint b = 2;
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            // Apply bloom to the texture color
            uint a = 8;
            a >>= 1;
            uint b = 2;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError as e:
        pytest.fail(f"Failed to parse ASSIGN_SHIFT_RIGHT token: {e}")


def test_logical_operators():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            // Using logical AND and logical OR operators
            float isLightOn;
            vec3 position = vec3(0.0);
            if ((position.x > 0.3 && position.z < 0.7) || position.y > 0.5) {
                isLightOn = 1.0;
            } else {
                isLightOn = 0.0;
            }
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            // Apply bloom to the texture color
            // Using logical AND and logical OR operators
            float isLightOn;
            vec3 position = vec3(0.0);
            if ((position.x > 0.3 && position.z < 0.7) || position.y > 0.5) {
                isLightOn = 1.0;
            } else {
                isLightOn = 0.0;
            }
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Logical operators not implemented.")


def test_var_assignment():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            vec4 testColor = vec4(1.0, 1.0, 1.0, 1.0);
            vec4 secondColor = vec4(0.5, 0.5, 0.5, 1.0);
            output.color = testColor + secondColor;
            vec2 p = vec2(1.0, 2.0);
            double noise = 0.5;
            double height = noise * 10.0;
            uint a = 1;
            uint b = 2;
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            vec2 p = vec2(1.0, 2.0);
            double noise = 0.5;
            double height = noise * 10.0;
            uint a = 1;
            uint b = 2;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Variable assignment parsing not implemented.")


def test_assign_ops():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            vec4 testColor = vec4(1.0, 1.0, 1.0, 1.0);
            vec4 secondColor = vec4(0.5, 0.5, 0.5, 1.0);
            output.color = testColor + secondColor;
            int xStatus = int(input.texCoord.x * 10.0);
            int yStatus = int(input.texCoord.y * 10.0);
            int zStatus = 5;
            int lightStatus = 0;

            xStatus |= yStatus;
            yStatus &= zStatus;
            zStatus %= xStatus;
            lightStatus = xStatus;
            lightStatus ^= zStatus;
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            int xStatus = int(input.color.x * 10.0);
            int yStatus = int(input.color.y * 10.0);
            int zStatus = 5;
            int lightStatus = 0;

            xStatus |= yStatus;
            yStatus &= zStatus;
            zStatus %= xStatus;
            lightStatus = xStatus;
            lightStatus ^= zStatus;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Assignment Operator parsing not implemented.")


def test_bitwise_operators():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            vec4 testColor = vec4(1.0, 1.0, 1.0, 1.0);
            float isLightOn = 2.0;
            int value = 2 >> 1;
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            int value = 1;
            value = value << 1;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise Shift not working")


def test_xor_operator():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            vec4 testColor = vec4(1.0, 1.0, 1.0, 1.0);
            vec2 vUV = vec2(0.0);
            vec3 position = vec3(0.0);
            vUV.x = float(int(position.x) ^ 3);  // XOR with 3
            vUV.y = float(int(position.y) ^ 5);  // XOR with 5
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            vec2 vUV = vec2(0.0);
            vec3 position = vec3(0.0);
            vUV.x = float(int(position.x) ^ 3);  // XOR with 3
            vUV.y = float(int(position.y) ^ 5);  // XOR with 5
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise XOR not working")


def test_and_operator():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    sampler2D iChannel0;
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            // Use bitwise AND on texture coordinates (for testing purposes)
            output.color = vec4(float(int(input.texCoord.x * 100.0) & 15), 
                                float(int(input.texCoord.y * 100.0) & 15), 
                                0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Simple fragment shader to display the result of the AND operation
            return vec4(input.color.rgb, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise AND not working")


def test_modulus_operations():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    sampler2D iChannel0;
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            // Test modulus operations
            int value = 1200;
            value = value % 10;      // Basic modulus
            value %= 5;              // Modulus assignment
            output.color = vec4(float(value) / 10.0, 0.0, 0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            return vec4(input.color.rgb, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operations not working")


def test_bitwise_not():
    code = """
    shader test {
        void main() {
            int a = 5;
            int b = ~a;  // Bitwise NOT
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        assert isinstance(ast, ShaderNode)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing failed")


def test_bitwise_expressions():
    code = """
    shader test {
        void main() {
            int a = 5;
            int b = ~a;  // Bitwise NOT
            int c = a & b;  // Bitwise AND
            int d = a | b;  // Bitwise OR
            int e = a ^ b;  // Bitwise XOR
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        assert isinstance(ast, ShaderNode)
    except SyntaxError:
        pytest.fail("Bitwise expressions parsing failed")


def test_array_syntax():
    """Test array declarations and array access syntax."""
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };
    
    struct Particle {
        vec3 position;
        vec3 velocity;
    };

    struct Material {
        float values[4];  // Fixed-size array
        vec3 colors[];    // Dynamic array
    };

    cbuffer Constants {
        float weights[8];
        int indices[10];
    };

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            
            // Array access in various forms
            float value = weights[2];
            int index = indices[5];
            
            // Array member access
            Material material;
            float x = material.values[0];
            vec3 color = material.colors[index];
            
            // Nested array access
            Particle particles[10];
            vec3 pos = particles[3].position;
            particles[index].velocity = vec3(1.0, 0.0, 0.0);
            
            // Array access in expressions
            float sum = weights[0] + weights[1] + weights[2];
            
            return output;
        }
    }
}
"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError as e:
        pytest.fail(f"Array syntax parsing failed: {e}")


def test_duplicate_cbuffer_members_fail_validation():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float value;
        };

        cbuffer Lighting {
            float value;
        };

        compute {
            void main() {
                float x = value;
            }
        }
    }
    """

    with pytest.raises(
        SyntaxError,
        match="Ambiguous cbuffer member name\\(s\\): value",
    ):
        parse_code(tokenize_code(code))


def test_duplicate_cbuffer_names_fail_validation():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float exposure;
        };

        cbuffer Camera {
            float gamma;
        };

        compute {
            void main() {
                float x = exposure + gamma;
            }
        }
    }
    """

    with pytest.raises(
        SyntaxError,
        match="Duplicate cbuffer name\\(s\\): Camera",
    ):
        parse_code(tokenize_code(code))


def test_cbuffer_name_conflicts_with_struct_fail_validation():
    code = """
    shader CBufferScope {
        struct Camera {
            float exposure;
        };

        cbuffer Camera {
            float gamma;
        };

        compute {
            void main() {
                float x = gamma;
            }
        }
    }
    """

    with pytest.raises(
        SyntaxError,
        match="Cbuffer name\\(s\\) conflict with existing declaration\\(s\\): Camera",
    ):
        parse_code(tokenize_code(code))


def test_cbuffer_member_conflicts_with_global_fail_validation():
    code = """
    shader CBufferScope {
        float exposure;

        cbuffer Camera {
            float exposure;
        };

        compute {
            void main() {
                float x = exposure;
            }
        }
    }
    """

    with pytest.raises(
        SyntaxError,
        match="Cbuffer member name\\(s\\) conflict with global declaration\\(s\\): exposure",
    ):
        parse_code(tokenize_code(code))


def test_array_parameter_syntax():
    code = """
    shader ArrayParams {
        float accumulate(float weights[4], vec3 colors[2]) {
            return weights[0] + colors[1].x;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert isinstance(function.parameters[0].param_type, ArrayType)
    assert isinstance(function.parameters[1].param_type, ArrayType)


def test_double_vector_type_keywords_parse():
    code = """
    shader DoubleVectors {
        compute {
            void main() {
                dvec2 precise = dvec2(1.0, 2.0);
                dvec4 weights = dvec4(1.0, 2.0, 3.0, 4.0);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    statements = ast.stages[ShaderStage.COMPUTE].entry_point.body.statements

    precise = statements[0]
    assert isinstance(precise.var_type, VectorType)
    assert isinstance(precise.var_type.element_type, PrimitiveType)
    assert precise.var_type.element_type.name == "double"
    assert precise.var_type.size == 2
    assert isinstance(precise.initial_value, FunctionCallNode)
    assert precise.initial_value.function.name == "dvec2"

    weights = statements[1]
    assert isinstance(weights.var_type, VectorType)
    assert weights.var_type.element_type.name == "double"
    assert weights.var_type.size == 4
    assert weights.initial_value.function.name == "dvec4"


def test_bool_vector_type_keywords_parse():
    code = """
    shader BoolVectors {
        compute {
            void main() {
                bvec2 mask = bvec2(true, false);
                bvec3 flags;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    statements = ast.stages[ShaderStage.COMPUTE].entry_point.body.statements

    mask = statements[0]
    assert isinstance(mask.var_type, VectorType)
    assert isinstance(mask.var_type.element_type, PrimitiveType)
    assert mask.var_type.element_type.name == "bool"
    assert mask.var_type.size == 2
    assert isinstance(mask.initial_value, FunctionCallNode)
    assert mask.initial_value.function.name == "bvec2"

    flags = statements[1]
    assert isinstance(flags.var_type, VectorType)
    assert flags.var_type.element_type.name == "bool"
    assert flags.var_type.size == 3


def test_generic_vector_type_keywords_parse():
    code = """
    shader GenericVectors {
        compute {
            void main() {
                vec2<f64> precise = vec2<f64>(1.0, 2.0);
                vec3<i32> index = vec3<i32>(1, 2, 3);
                vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                vec2<bool> flags = vec2<bool>(true, false);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    statements = ast.stages[ShaderStage.COMPUTE].entry_point.body.statements

    expected = [
        ("precise", "double", 2, "vec2<f64>"),
        ("index", "int", 3, "vec3<i32>"),
        ("mask", "uint", 4, "vec4<u32>"),
        ("flags", "bool", 2, "vec2<bool>"),
    ]
    for statement, (name, element_type, size, constructor) in zip(statements, expected):
        assert statement.name == name
        assert isinstance(statement.var_type, VectorType)
        assert statement.var_type.element_type.name == element_type
        assert statement.var_type.size == size
        assert isinstance(statement.initial_value, FunctionCallNode)
        assert statement.initial_value.function.name == constructor


def test_matrix_type_keywords_parse():
    code = """
    shader Matrices {
        compute {
            void main() {
                mat3x4 affine;
                dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                dmat4x3 jacobian;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    statements = ast.stages[ShaderStage.COMPUTE].entry_point.body.statements

    affine = statements[0]
    assert isinstance(affine.var_type, MatrixType)
    assert affine.var_type.element_type.name == "float"
    assert affine.var_type.rows == 3
    assert affine.var_type.cols == 4

    precise = statements[1]
    assert isinstance(precise.var_type, MatrixType)
    assert precise.var_type.element_type.name == "double"
    assert precise.var_type.rows == 2
    assert precise.var_type.cols == 2
    assert isinstance(precise.initial_value, FunctionCallNode)
    assert precise.initial_value.function.name == "dmat2"

    jacobian = statements[2]
    assert isinstance(jacobian.var_type, MatrixType)
    assert jacobian.var_type.element_type.name == "double"
    assert jacobian.var_type.rows == 4
    assert jacobian.var_type.cols == 3


def test_sampler_type_keywords_parse():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray colorArray;
        samplercube environmentMap;
        sampler2dshadow shadowMap;
        sampler2darrayshadow cascades;
        samplercubeshadow pointShadowMap;
        samplercubearray reflectionProbes;
        samplercubearrayshadow shadowProbes;
        sampler2dms colorMs;
        sampler2dmsarray colorMsArray;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    resource_types = [var.var_type.name for var in ast.global_variables]

    assert resource_types == [
        "sampler",
        "sampler2D",
        "sampler2DArray",
        "samplerCube",
        "sampler2DShadow",
        "sampler2DArrayShadow",
        "samplerCubeShadow",
        "samplerCubeArray",
        "samplerCubeArrayShadow",
        "sampler2DMS",
        "sampler2DMSArray",
    ]


def test_image_type_keywords_parse():
    code = """
    shader Resources {
        iimage2d signedImage;
        iimage3d signedVolume;
        iimage2darray signedLayers;
        iimage2dms signedMs;
        iimage2dmsarray signedMsLayers;
        uimage2d unsignedImage;
        uimage3d unsignedVolume;
        uimage2darray unsignedLayers;
        uimage2dms unsignedMs;
        uimage2dmsarray unsignedMsLayers;
        image2D outputImage;
        image3D volumeImage;
        imageCube cubeImage;
        image2DArray layerImage;
        image2DMS outputMs;
        image2DMSArray layerMs;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    resource_types = [var.var_type.name for var in ast.global_variables]

    assert resource_types == [
        "iimage2D",
        "iimage3D",
        "iimage2DArray",
        "iimage2DMS",
        "iimage2DMSArray",
        "uimage2D",
        "uimage3D",
        "uimage2DArray",
        "uimage2DMS",
        "uimage2DMSArray",
        "image2D",
        "image3D",
        "imageCube",
        "image2DArray",
        "image2DMS",
        "image2DMSArray",
    ]


def test_image_format_attributes_parse():
    code = """
    shader Resources {
        image2D halfColor @rgba16f;
        uimage2D counters @ r32ui;
        image2D normalizedColor @format(rgba8);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [var.name for var in ast.global_variables] == [
        "halfColor",
        "counters",
        "normalizedColor",
    ]
    assert [attr.name for attr in ast.global_variables[0].attributes] == ["rgba16f"]
    assert [attr.name for attr in ast.global_variables[1].attributes] == ["r32ui"]
    assert [attr.name for attr in ast.global_variables[2].attributes] == ["format"]
    assert ast.global_variables[2].attributes[0].arguments[0].name == "rgba8"


def test_image_format_parameter_attributes_parse():
    code = """
    shader Resources {
        float touchImages(
            image2D scalarImage @r32f,
            image3D signedImage @ r32i,
            image2DArray unsignedImage @format(r32ui)
        ) {
            return 0.0;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    params = ast.functions[0].parameters

    assert [param.name for param in params] == [
        "scalarImage",
        "signedImage",
        "unsignedImage",
    ]
    assert [attr.name for attr in params[0].attributes] == ["r32f"]
    assert [attr.name for attr in params[1].attributes] == ["r32i"]
    assert [attr.name for attr in params[2].attributes] == ["format"]
    assert params[2].attributes[0].arguments[0].name == "r32ui"


def test_extended_shader_stages_parse():
    code = """
    shader main {
        task {
            void main() { return; }
        }
        mesh {
            void main() { return; }
        }
        tessellation_control {
            void main() { return; }
        }
        tessellation_evaluation {
            void main() { return; }
        }
        ray_generation {
            void main() { return; }
        }
        ray_any_hit {
            void main() { return; }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        assert ShaderStage.TASK in ast.stages
        assert ShaderStage.MESH in ast.stages
        assert ShaderStage.TESSELLATION_CONTROL in ast.stages
        assert ShaderStage.TESSELLATION_EVALUATION in ast.stages
        assert ShaderStage.RAY_GENERATION in ast.stages
        assert ShaderStage.RAY_ANY_HIT in ast.stages
    except SyntaxError as e:
        pytest.fail(f"Extended shader stage parsing failed: {e}")


def test_wave_intrinsic_parses_to_node():
    code = """
    shader main {
        compute {
            void main() {
                int sum = WaveActiveSum(1);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    stage = ast.stages[ShaderStage.COMPUTE]
    body = stage.entry_point.body.statements
    var_decl = next(stmt for stmt in body if isinstance(stmt, VariableNode))
    assert isinstance(var_decl.initial_value, WaveOpNode)


def test_raytracing_intrinsic_parses_to_node():
    code = """
    shader main {
        ray_generation {
            void main() {
                int handle = TraceRay(1, 2, 3);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    stage = ast.stages[ShaderStage.RAY_GENERATION]
    body = stage.entry_point.body.statements
    var_decl = next(stmt for stmt in body if isinstance(stmt, VariableNode))
    assert isinstance(var_decl.initial_value, RayTracingOpNode)


def test_mesh_intrinsic_parses_to_node():
    code = """
    shader main {
        mesh {
            void main() {
                int ok = SetMeshOutputCounts(1, 1);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    stage = ast.stages[ShaderStage.MESH]
    body = stage.entry_point.body.statements
    var_decl = next(stmt for stmt in body if isinstance(stmt, VariableNode))
    assert isinstance(var_decl.initial_value, MeshOpNode)


def test_rayquery_method_parses_to_node():
    code = """
    shader main {
        compute {
            void main() {
                int ok = rayQuery.Proceed();
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    stage = ast.stages[ShaderStage.COMPUTE]
    body = stage.entry_point.body.statements
    var_decl = next(stmt for stmt in body if isinstance(stmt, VariableNode))
    assert isinstance(var_decl.initial_value, RayQueryOpNode)


def test_unsigned_integer_literals_parse_as_uint_nodes():
    code = """
    shader main {
        compute {
            void main() {
                uint value = pack(7u, 0xFu, 0b101U, 0o17u);
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    stage = ast.stages[ShaderStage.COMPUTE]
    body = stage.entry_point.body.statements
    var_decl = next(stmt for stmt in body if isinstance(stmt, VariableNode))
    args = var_decl.initial_value.args

    assert len(args) == 4
    assert [arg.value for arg in args] == [7, 15, 5, 15]
    assert all(isinstance(arg, LiteralNode) for arg in args)
    assert all(arg.literal_type.name == "uint" for arg in args)


if __name__ == "__main__":
    pytest.main()

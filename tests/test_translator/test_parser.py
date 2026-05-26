from crosstl.translator.lexer import Lexer
import pytest
from pathlib import Path
from typing import List
from crosstl.translator.parser import Parser
from crosstl.translator.ast import (
    ArrayType,
    AssignmentNode,
    ConstructorNode,
    ConstructorPatternNode,
    DoWhileNode,
    ForInNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    LiteralNode,
    LoopNode,
    MatrixType,
    MemberAccessNode,
    NamedType,
    PointerType,
    PreprocessorNode,
    PrimitiveType,
    RangeNode,
    ReferenceType,
    ShaderNode,
    ShaderStage,
    StructPatternNode,
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


def test_do_while_statement_parses_to_do_while_node():
    code = """
    shader main {
        int helper(int limit) {
            int i = 0;
            do {
                i = i + 1;
            } while (i < limit);
            return i;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    helper = ast.functions[0]

    assert any(isinstance(stmt, DoWhileNode) for stmt in helper.body.statements)


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


def test_compound_assignments_parse_operator_tokens():
    code = """
    shader CompoundAssignments {
        compute {
            void main() {
                float value = 1.0;
                value += 2.0;
                value -= 3.0;
                value *= 4.0;
                value /= 5.0;
                int bits = 7;
                bits &= 3;
                bits |= 8;
                bits ^= 2;
                bits <<= 1;
                bits >>= 1;
                bits %= 2;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    statements = ast.stages[ShaderStage.COMPUTE].entry_point.body.statements
    assignments = [
        stmt.expression
        for stmt in statements
        if isinstance(getattr(stmt, "expression", None), AssignmentNode)
    ]

    assert [assignment.operator for assignment in assignments] == [
        "+=",
        "-=",
        "*=",
        "/=",
        "&=",
        "|=",
        "^=",
        "<<=",
        ">>=",
        "%=",
    ]


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


def test_nested_cbuffer_array_members_parse_as_nested_array_types():
    code = """
    shader NestedCBufferArrays {
        cbuffer Constants {
            int values[2][3];
            uint offsets[4][5];
        };
    }
    """

    ast = parse_code(tokenize_code(code))
    cbuffer = ast.cbuffers[0]
    values = cbuffer.members[0].member_type
    offsets = cbuffer.members[1].member_type

    assert isinstance(values, ArrayType)
    assert isinstance(values.element_type, ArrayType)
    assert values.size.value == 3
    assert values.element_type.size.value == 2
    assert values.element_type.element_type.name == "int"

    assert isinstance(offsets, ArrayType)
    assert isinstance(offsets.element_type, ArrayType)
    assert offsets.size.value == 5
    assert offsets.element_type.size.value == 4
    assert offsets.element_type.element_type.name == "uint"


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


def test_resource_parameter_syntax():
    code = """
    @compute
    @workgroup_size(1, 1, 1)
    fn kernel(
        @group(0) @binding(0) var<storage, read_write> data: array<f32>,
        @group(0) @binding(1) var<storage, read> indices: array<i32>,
        f32 value
    ) {
        data[indices[0]] = value;
    }
    """

    ast = parse_code(tokenize_code(code))
    function = ast.functions[0]
    data, indices, value = function.parameters

    assert function.name == "kernel"
    assert isinstance(function.return_type, PrimitiveType)
    assert function.return_type.name == "void"
    assert [attribute.name for attribute in function.attributes] == [
        "compute",
        "workgroup_size",
    ]

    assert data.name == "data"
    assert isinstance(data.param_type, ArrayType)
    assert isinstance(data.param_type.element_type, PrimitiveType)
    assert data.param_type.element_type.name == "f32"
    assert data.param_type.size is None
    assert getattr(data, "resource_qualifiers") == ["storage", "read_write"]
    assert [attribute.name for attribute in data.attributes] == ["group", "binding"]

    assert indices.name == "indices"
    assert isinstance(indices.param_type, ArrayType)
    assert indices.param_type.element_type.name == "i32"
    assert getattr(indices, "resource_qualifiers") == ["storage", "read"]

    assert value.name == "value"
    assert isinstance(value.param_type, PrimitiveType)
    assert value.param_type.name == "f32"


def test_compute_layout_execution_config_parsing():
    code = """
    shader ComputeLayout {
        const int GROUP_SIZE = 8;
        compute {
            layout(local_size_x = GROUP_SIZE * 2, local_size_y = 8, local_size_z = 1) in;
            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    compute_stage = ast.stages[ShaderStage.COMPUTE]

    assert compute_stage.execution_config == {
        "local_size_x": "GROUP_SIZE * 2",
        "local_size_y": "8",
        "local_size_z": "1",
    }
    assert len(compute_stage.layout_qualifiers) == 1
    layout = compute_stage.layout_qualifiers[0]
    assert layout.direction == "in"
    assert [entry.name for entry in layout.entries] == [
        "local_size_x",
        "local_size_y",
        "local_size_z",
    ]
    assert layout.entries[0].arguments[0].op == "*"
    assert [entry.arguments[0].value for entry in layout.entries[1:]] == [8, 1]


def test_compute_layout_does_not_consume_resource_layouts():
    code = """
    shader ComputeLayoutWithResources {
        compute {
            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
            layout(set = 0, binding = 0) buffer float values[];
            layout(set = 0, binding = 1) uniform sampler2D sourceTexture;
            layout(set = 0, binding = 2) sampler sourceSampler;

            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    compute_stage = ast.stages[ShaderStage.COMPUTE]

    assert compute_stage.execution_config == {
        "local_size_x": "8",
        "local_size_y": "8",
        "local_size_z": "1",
    }
    assert len(compute_stage.layout_qualifiers) == 1
    assert [variable.name for variable in compute_stage.local_variables] == [
        "values",
        "sourceTexture",
        "sourceSampler",
    ]
    assert [
        attribute.name for attribute in compute_stage.local_variables[0].attributes
    ] == [
        "set",
        "binding",
    ]
    assert [
        attribute.arguments[0].value
        for attribute in compute_stage.local_variables[0].attributes
    ] == [0, 0]
    assert [
        attribute.name for attribute in compute_stage.local_variables[1].attributes
    ] == [
        "set",
        "binding",
    ]
    assert compute_stage.local_variables[1].attributes[1].arguments[0].value == 1


def test_global_resource_layout_attributes_are_preserved():
    code = """
    shader GlobalResourceLayouts {
        layout(set = 2, binding = 7) uniform sampler2D sourceTexture;
        layout(binding = 3) sampler sourceSampler;

        fragment {
            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    source_texture, source_sampler = ast.global_variables

    assert source_texture.name == "sourceTexture"
    assert [attribute.name for attribute in source_texture.attributes] == [
        "set",
        "binding",
    ]
    assert [
        attribute.arguments[0].value for attribute in source_texture.attributes
    ] == [
        2,
        7,
    ]

    assert source_sampler.name == "sourceSampler"
    assert [attribute.name for attribute in source_sampler.attributes] == ["binding"]
    assert source_sampler.attributes[0].arguments[0].value == 3


def test_translation_unit_resource_layout_attributes_are_preserved():
    code = """
    layout(set = 1, binding = 4) uniform sampler2D sourceTexture;

    shader TranslationUnitResourceLayout {
        fragment {
            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    source_texture = ast.global_variables[0]

    assert ast.name == "TranslationUnitResourceLayout"
    assert source_texture.name == "sourceTexture"
    assert source_texture.qualifiers == ["uniform"]
    assert [attribute.name for attribute in source_texture.attributes] == [
        "set",
        "binding",
    ]
    assert [
        attribute.arguments[0].value for attribute in source_texture.attributes
    ] == [
        1,
        4,
    ]


def test_stage_interface_layout_qualifiers_are_preserved_on_variables():
    code = """
    shader StageInterfaceLayouts {
        vertex {
            layout(location = 0) flat in vec3 position;
            layout(location = 1) noperspective centroid out vec2 texCoord;
            threadgroup float scratch[64];

            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    vertex_stage = ast.stages[ShaderStage.VERTEX]
    position, tex_coord, scratch = vertex_stage.local_variables

    assert position.name == "position"
    assert position.qualifiers == ["flat", "in"]
    assert [attribute.name for attribute in position.attributes] == ["location"]
    assert position.attributes[0].arguments[0].value == 0

    assert tex_coord.name == "texCoord"
    assert tex_coord.qualifiers == ["noperspective", "centroid", "out"]
    assert [attribute.name for attribute in tex_coord.attributes] == ["location"]
    assert tex_coord.attributes[0].arguments[0].value == 1

    assert scratch.name == "scratch"
    assert scratch.qualifiers == ["threadgroup"]
    assert isinstance(scratch.var_type, ArrayType)
    assert scratch.var_type.size.value == 64


def test_resource_layout_preserves_access_and_memory_qualifiers():
    code = """
    shader ResourceQualifierLayouts {
        layout(binding = 0, r32ui) coherent readonly uniform uimage2D counters;
        layout(binding = 1, rgba32f) writeonly uniform image2D outImage;

        compute {
            layout(std430, binding = 2) readonly buffer float values[];
            device float deviceValue;
            constant int lookup;

            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    counters, out_image = ast.global_variables
    compute_stage = ast.stages[ShaderStage.COMPUTE]
    values, device_value, lookup = compute_stage.local_variables

    assert counters.name == "counters"
    assert counters.qualifiers == ["coherent", "readonly", "uniform"]
    assert [attribute.name for attribute in counters.attributes] == [
        "binding",
        "r32ui",
    ]

    assert out_image.name == "outImage"
    assert out_image.qualifiers == ["writeonly", "uniform"]
    assert [attribute.name for attribute in out_image.attributes] == [
        "binding",
        "rgba32f",
    ]

    assert values.name == "values"
    assert values.qualifiers == ["readonly", "buffer"]
    assert [attribute.name for attribute in values.attributes] == [
        "std430",
        "binding",
    ]
    assert isinstance(values.var_type, ArrayType)

    assert device_value.qualifiers == ["device"]
    assert lookup.qualifiers == ["constant"]


def test_glsl_layout_buffer_block_lowers_to_struct_and_resource_variable():
    code = """
    shader BufferBlockLayouts {
        layout(std430, set = 1, binding = 2) readonly buffer ParticleBlock {
            vec4 position;
            uint count;
        } particles[4];
    }
    """

    ast = parse_code(tokenize_code(code))
    block_struct = ast.structs[0]
    particles = ast.global_variables[0]

    assert block_struct.name == "ParticleBlock"
    assert [member.name for member in block_struct.members] == ["position", "count"]

    assert particles.name == "particles"
    assert particles.qualifiers == ["readonly", "buffer"]
    assert isinstance(particles.var_type, ArrayType)
    assert particles.var_type.element_type.name == "ParticleBlock"
    assert particles.var_type.size.value == 4
    assert [attribute.name for attribute in particles.attributes] == [
        "glsl_buffer_block",
        "set",
        "binding",
    ]
    assert particles.attributes[0].arguments[0].name == "std430"
    assert [attribute.arguments[0].value for attribute in particles.attributes[1:]] == [
        1,
        2,
    ]


def test_translation_unit_glsl_layout_buffer_block_lowers_to_ir():
    code = """
    layout(std140, binding = 3) buffer Globals {
        mat4 transform;
    };

    shader BufferBlockTranslationUnit {
        compute {
            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    block_struct = ast.structs[0]
    globals_block = ast.global_variables[0]

    assert block_struct.name == "Globals"
    assert block_struct.members[0].name == "transform"
    assert globals_block.name == "globals"
    assert globals_block.var_type.name == "Globals"
    assert globals_block.qualifiers == ["buffer"]
    assert [attribute.name for attribute in globals_block.attributes] == [
        "glsl_buffer_block",
        "binding",
    ]
    assert globals_block.attributes[0].arguments[0].name == "std140"
    assert globals_block.attributes[1].arguments[0].value == 3


def test_capitalized_type_names_are_not_mistaken_for_qualifiers():
    code = """
    shader CapitalizedTypeNames {
        struct Out {
            vec4 color;
        };

        vertex {
            Out main() {
                Out output;
                return output;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    entry_point = ast.stages[ShaderStage.VERTEX].entry_point

    assert entry_point.return_type.name == "Out"
    assert len(entry_point.body.statements) == 2
    assert entry_point.body.statements[0].name == "output"
    assert entry_point.body.statements[0].var_type.name == "Out"


def test_stage_layout_qualifiers_preserve_geometry_and_tessellation_metadata():
    code = """
    shader StageLayouts {
        geometry {
            layout(triangles) in;
            layout(triangle_strip, max_vertices = 3) out;
            void main() {
            }
        }

        tessellation_control {
            layout(vertices = 4) out;
            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    geometry_stage = ast.stages[ShaderStage.GEOMETRY]
    tessellation_stage = ast.stages[ShaderStage.TESSELLATION_CONTROL]

    assert [layout.direction for layout in geometry_stage.layout_qualifiers] == [
        "in",
        "out",
    ]
    assert [entry.name for entry in geometry_stage.layout_qualifiers[0].entries] == [
        "triangles"
    ]
    output_layout = geometry_stage.layout_qualifiers[1]
    assert [entry.name for entry in output_layout.entries] == [
        "triangle_strip",
        "max_vertices",
    ]
    assert output_layout.entries[1].arguments[0].value == 3

    assert tessellation_stage.layout_qualifiers[0].direction == "out"
    assert tessellation_stage.layout_qualifiers[0].entries[0].name == "vertices"
    assert tessellation_stage.layout_qualifiers[0].entries[0].arguments[0].value == 4


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            compute {
                layout(local_size_x = 8) in;
                layout(local_size_x = 16) in;
                void main() {
                }
            }
            """,
            "Conflicting stage layout 'local_size_x'",
        ),
        (
            """
            compute {
                layout(local_size_x = 8) out;
                void main() {
                }
            }
            """,
            "local_size_x.*must use 'in' direction",
        ),
        (
            """
            geometry {
                layout(triangles) in;
                layout(lines) in;
                void main() {
                }
            }
            """,
            "Conflicting stage layout entries.*lines.*triangles",
        ),
        (
            """
            tessellation_evaluation {
                layout(triangles, quads) in;
                void main() {
                }
            }
            """,
            "Conflicting stage layout entries.*quads.*triangles",
        ),
        (
            """
            tessellation_evaluation {
                layout(equal_spacing, fractional_even_spacing) in;
                void main() {
                }
            }
            """,
            "Conflicting stage layout entries.*equal_spacing.*fractional_even_spacing",
        ),
        (
            """
            tessellation_evaluation {
                layout(cw, ccw) in;
                void main() {
                }
            }
            """,
            "Conflicting stage layout entries.*ccw.*cw",
        ),
    ],
)
def test_conflicting_stage_layout_metadata_fails_validation(stage_source, message):
    code = f"""
    shader ConflictingStageLayouts {{
        {stage_source}
    }}
    """

    with pytest.raises(ValueError, match=message):
        parse_code(tokenize_code(code))


def test_duplicate_matching_stage_layout_metadata_values_are_allowed():
    code = """
    shader MatchingStageLayouts {
        compute {
            layout(local_size_x = GROUP_SIZE * 2) in;
            layout(local_size_x = GROUP_SIZE * 2, local_size_y = 4) in;
            void main() {
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    compute_stage = ast.stages[ShaderStage.COMPUTE]

    assert compute_stage.execution_config == {
        "local_size_x": "GROUP_SIZE * 2",
        "local_size_y": "4",
    }
    assert len(compute_stage.layout_qualifiers) == 2


def test_lambda_call_preserves_typed_parameters_and_block_body_parse():
    code = """
    shader LambdaBlocks {
        void closure_blocks(Values values) {
            let mapped = map(
                values,
                lambda(Result<i32, i32> value, {
                    prepare(value);
                    return Ok(value);
                })
            );
            let less = lambda(x, (x < 1));
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    statements = ast.functions[0].body.statements
    mapped = statements[0]
    map_call = mapped.initial_value

    assert isinstance(map_call, FunctionCallNode)
    assert map_call.function.name == "map"

    lambda_call = map_call.arguments[1]
    assert isinstance(lambda_call, FunctionCallNode)
    assert lambda_call.function.name == "lambda"
    assert all(isinstance(arg, IdentifierNode) for arg in lambda_call.arguments)
    assert [arg.name for arg in lambda_call.arguments] == [
        "Result<i32, i32> value",
        "{ prepare(value); return Ok(value); }",
    ]

    less_call = statements[1].initial_value
    assert isinstance(less_call, FunctionCallNode)
    assert less_call.function.name == "lambda"
    assert [arg.name for arg in less_call.arguments] == ["x", "(x < 1)"]


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


def test_literal_generic_type_arguments_parse():
    code = """
    shader GenericLiteralArgs {
        struct HSInput {
            vec3 position;
        };

        void consume(InputPatch<HSInput, 3> patch) { }
    }
    """

    ast = parse_code(tokenize_code(code))
    param_type = ast.functions[0].parameters[0].param_type

    assert isinstance(param_type, NamedType)
    assert param_type.name == "InputPatch"
    assert isinstance(param_type.generic_args[0], NamedType)
    assert param_type.generic_args[0].name == "HSInput"
    assert isinstance(param_type.generic_args[1], LiteralNode)
    assert param_type.generic_args[1].value == 3


def test_hlsl_parameter_qualifiers_parse():
    code = """
    shader HLSLParameterQualifiers {
        struct GSInput {
            vec3 position;
        };

        struct GSOutput {
            vec4 position;
        };

        void consume(
            triangle GSInput input[3],
            inout TriangleStream<GSOutput> stream,
            const OutputPatch<GSOutput, 3> patch
        ) { }
    }
    """

    ast = parse_code(tokenize_code(code))
    input_param, stream_param, patch_param = ast.functions[0].parameters

    assert input_param.qualifiers == ["triangle"]
    assert isinstance(input_param.param_type, ArrayType)
    assert input_param.param_type.element_type.name == "GSInput"
    assert input_param.param_type.size.value == 3

    assert stream_param.qualifiers == ["inout"]
    assert isinstance(stream_param.param_type, NamedType)
    assert stream_param.param_type.name == "TriangleStream"
    assert stream_param.param_type.generic_args[0].name == "GSOutput"

    assert patch_param.qualifiers == ["const"]
    assert isinstance(patch_param.param_type, NamedType)
    assert patch_param.param_type.name == "OutputPatch"
    assert patch_param.param_type.generic_args[0].name == "GSOutput"
    assert patch_param.param_type.generic_args[1].value == 3


def test_backend_parameter_address_and_access_qualifiers_parse():
    code = """
    shader BackendParameterQualifiers {
        struct Payload {
            float value;
        };

        void consume(
            threadgroup Payload* payload @payload,
            constant Payload& payloadRef,
            device Payload& mut mutablePayloadRef,
            device float values[],
            constant int count,
            readonly image2D source @r32f,
            writeonly image2D target @rgba32f
        ) { }
    }
    """

    ast = parse_code(tokenize_code(code))
    (
        payload,
        payload_ref,
        mutable_payload_ref,
        values,
        count,
        source,
        target,
    ) = ast.functions[0].parameters

    assert payload.qualifiers == ["threadgroup"]
    assert payload.name == "payload"
    assert isinstance(payload.param_type, PointerType)
    assert payload.param_type.pointee_type.name == "Payload"
    assert payload.attributes[0].name == "payload"

    assert payload_ref.qualifiers == ["constant"]
    assert isinstance(payload_ref.param_type, ReferenceType)
    assert payload_ref.param_type.referenced_type.name == "Payload"
    assert payload_ref.param_type.is_mutable is False

    assert mutable_payload_ref.qualifiers == ["device"]
    assert isinstance(mutable_payload_ref.param_type, ReferenceType)
    assert mutable_payload_ref.param_type.referenced_type.name == "Payload"
    assert mutable_payload_ref.param_type.is_mutable is True

    assert values.qualifiers == ["device"]
    assert isinstance(values.param_type, ArrayType)

    assert count.qualifiers == ["constant"]
    assert count.param_type.name == "int"

    assert source.qualifiers == ["readonly"]
    assert source.attributes[0].name == "r32f"

    assert target.qualifiers == ["writeonly"]
    assert target.attributes[0].name == "rgba32f"


def test_address_space_alias_qualifiers_are_allowed():
    code = """
    shader AddressSpaceAliases {
        shared threadgroup float workgroupScratch;
        groupshared workgroup int groupCounter;
        global device float deviceValue;
        local private float localValue;
        function thread int threadValue;

        void consume(global device float* data, shared threadgroup float* scratch) {
        }
    }
    """

    ast = parse_code(tokenize_code(code))

    assert ast.global_variables[0].qualifiers == ["shared", "threadgroup"]
    assert ast.global_variables[1].qualifiers == ["groupshared", "workgroup"]
    assert ast.global_variables[2].qualifiers == ["global", "device"]
    assert ast.global_variables[3].qualifiers == ["local", "private"]
    assert ast.global_variables[4].qualifiers == ["function", "thread"]
    data, scratch = ast.functions[0].parameters
    assert data.qualifiers == ["global", "device"]
    assert scratch.qualifiers == ["shared", "threadgroup"]


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            "device threadgroup float scratch;",
            "Conflicting address space metadata.*@device.*@threadgroup",
        ),
        (
            "constant global float value;",
            "Conflicting address space metadata.*@constant.*@global",
        ),
        (
            "storage private float value;",
            "Conflicting address space metadata.*@private.*@storage",
        ),
    ],
)
def test_conflicting_variable_address_space_metadata_fails_validation(
    declaration, message
):
    code = f"""
    shader ConflictingVariableAddressSpaces {{
        {declaration}
    }}
    """

    with pytest.raises(ValueError, match=message):
        parse_code(tokenize_code(code))


def test_conflicting_parameter_address_space_metadata_fails_validation():
    code = """
    shader ConflictingParameterAddressSpaces {
        void consume(device threadgroup float* scratch) {
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting address space metadata.*@device.*@threadgroup",
    ):
        parse_code(tokenize_code(code))


def test_shader_generic_struct_wrapper_parses_without_leaking_members(capsys):
    code = """
    shader GenericWrapper {
        generic<T, E> struct Result {
            value: T;
            error: E;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    captured = capsys.readouterr()

    assert "Parser may be stuck" not in captured.out
    assert ast.name == "GenericWrapper"
    assert ast.global_variables == []
    result_struct = next(struct for struct in ast.structs if struct.name == "Result")
    assert [param.name for param in result_struct.generic_params] == ["T", "E"]
    assert [member.name for member in result_struct.members] == ["value", "error"]


def test_shader_generic_function_signature_parses_rust_style(capsys):
    code = """
    shader GenericFunctions {
        generic<T> fn identity(value: T) -> T {
            return value;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    captured = capsys.readouterr()

    assert "Parser may be stuck" not in captured.out
    function = next(func for func in ast.functions if func.name == "identity")
    assert [param.name for param in function.generic_params] == ["T"]
    assert function.parameters[0].name == "value"
    assert function.parameters[0].param_type.name == "T"
    assert function.return_type.name == "T"


def test_shader_traits_and_generic_constraints_parse(capsys):
    code = """
    shader GenericBounds {
        trait Numeric {
            fn zero() -> Self;
            fn add(self, other: Self) -> Self;
        }

        struct ScalarBox<T: Numeric> {
            value: T;
        }

        generic<T: Numeric + Copy> fn add_zero(value: T) -> T {
            return value.add(T::zero());
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    captured = capsys.readouterr()

    assert "Parser may be stuck" not in captured.out
    trait = next(struct for struct in ast.structs if struct.name == "Numeric")
    assert getattr(trait, "is_trait", False) is True
    assert [method.name for method in trait.members] == ["zero", "add"]

    scalar_box = next(struct for struct in ast.structs if struct.name == "ScalarBox")
    assert scalar_box.generic_params[0].name == "T"
    assert [
        constraint.name for constraint in scalar_box.generic_params[0].constraints
    ] == ["Numeric"]

    function = next(func for func in ast.functions if func.name == "add_zero")
    assert function.generic_params[0].name == "T"
    assert [
        constraint.name for constraint in function.generic_params[0].constraints
    ] == [
        "Numeric",
        "Copy",
    ]


def test_generic_pattern_matching_example_does_not_trigger_parser_stuck_warning(
    capsys,
):
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "examples/advanced/GenericPatternMatching.cgl").read_text()

    ast = parse_code(tokenize_code(source))
    captured = capsys.readouterr()

    assert "Parser may be stuck" not in captured.out
    vector_operation = next(
        func for func in ast.functions if func.name == "vector_operation"
    )
    assert vector_operation.body.statements
    assert vector_operation.body.statements[0].__class__.__name__ == "MatchNode"
    process_lighting_model = next(
        func for func in ast.functions if func.name == "process_lighting_model"
    )
    assert len(process_lighting_model.body.statements) == 1
    lighting_match = process_lighting_model.body.statements[0]
    assert lighting_match.__class__.__name__ == "MatchNode"
    assert len(lighting_match.arms) == 4
    vertex_entry = ast.stages[ShaderStage.VERTEX].entry_point
    processed_normal = next(
        stmt for stmt in vertex_entry.body.statements if stmt.name == "processed_normal"
    )
    assert processed_normal.initial_value.__class__.__name__ == "MatchNode"
    assert len(processed_normal.initial_value.arms) == 2
    assert [struct.name for struct in ast.stages[ShaderStage.VERTEX].local_structs] == [
        "VertexInput",
        "VertexOutput",
    ]
    vertex_input = ast.stages[ShaderStage.VERTEX].local_structs[0]
    assert [member.name for member in vertex_input.members] == [
        "position",
        "normal",
        "uv",
        "color",
    ]
    fragment_input = ast.stages[ShaderStage.FRAGMENT].local_structs[0]
    assert [member.name for member in fragment_input.members] == [
        "world_position",
        "normal",
        "uv",
        "color",
    ]


def test_match_path_constructor_and_struct_patterns_parse():
    code = """
    shader PatternCases {
        fn inspect(result: Result<int, Error>, model: LightingModel) -> int {
            match result {
                Result::Ok(value) => value,
                Result::Err(_) => 0,
            }

            match model {
                LightingModel::Phong { ambient, diffuse, specular, shininess } if shininess > 0.0 => 1,
                LightingModel::Toon { base_color, .. } => 2,
            }

            return 0;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    function = ast.functions[0]
    result_match = function.body.statements[0]
    model_match = function.body.statements[1]

    ok_pattern = result_match.arms[0].pattern
    assert isinstance(ok_pattern, ConstructorPatternNode)
    assert ok_pattern.type_name == "Result::Ok"
    assert isinstance(ok_pattern.arguments[0], IdentifierPatternNode)
    assert ok_pattern.arguments[0].name == "value"

    err_pattern = result_match.arms[1].pattern
    assert isinstance(err_pattern, ConstructorPatternNode)
    assert err_pattern.type_name == "Result::Err"
    assert err_pattern.arguments[0].__class__.__name__ == "WildcardPatternNode"

    phong_pattern = model_match.arms[0].pattern
    assert isinstance(phong_pattern, StructPatternNode)
    assert phong_pattern.type_name == "LightingModel::Phong"
    assert list(phong_pattern.field_patterns) == [
        "ambient",
        "diffuse",
        "specular",
        "shininess",
    ]
    assert model_match.arms[0].guard.operator == ">"

    toon_pattern = model_match.arms[1].pattern
    assert isinstance(toon_pattern, StructPatternNode)
    assert toon_pattern.type_name == "LightingModel::Toon"
    assert list(toon_pattern.field_patterns) == ["base_color"]
    assert toon_pattern.has_rest is True


def test_match_arm_block_tail_expression_parses_without_semicolon():
    code = """
    shader PatternCases {
        fn convert(op: VectorOp) -> Result<int, MathError> {
            match op {
                VectorOp::Cross => {
                    Result::Ok(1)
                },
                _ => Result::Err(MathError::InvalidInput),
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    match_stmt = ast.functions[0].body.statements[0]
    cross_body = match_stmt.arms[0].body
    cross_tail = cross_body.statements[-1]
    fallback_body = match_stmt.arms[1].body

    assert cross_tail.__class__.__name__ == "ExpressionStatementNode"
    assert cross_tail.is_tail_expression is True
    assert isinstance(cross_tail.expression, FunctionCallNode)
    assert cross_tail.expression.function.name == "Result::Ok"
    assert fallback_body.is_tail_expression is True


def test_method_calls_and_shorthand_path_constructors_parse():
    code = """
    shader ConstructorCases {
        fn build(v1: Vec3<T>, v2: Vec3<T>, color: vec4, depth: float, op: VectorOp) -> int {
            match op {
                VectorOp::Add => Result::Ok(Vec3 { x: v1.x.add(v2.x), y: v1.y.mul(v2.y), z: v1.z }),
                _ => Result::Ok(RenderOutput::Clear { color, depth }),
            }

            return 0;
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    match_stmt = ast.functions[0].body.statements[0]
    add_call = match_stmt.arms[0].body.expression
    clear_call = match_stmt.arms[1].body.expression

    vec_constructor = add_call.arguments[0]
    assert isinstance(vec_constructor, ConstructorNode)
    assert list(vec_constructor.named_arguments) == ["x", "y", "z"]

    x_call = vec_constructor.named_arguments["x"]
    assert isinstance(x_call, FunctionCallNode)
    assert isinstance(x_call.function, MemberAccessNode)
    assert x_call.function.member == "add"
    assert isinstance(x_call.function.object_expr, MemberAccessNode)
    assert x_call.function.object_expr.member == "x"

    clear_constructor = clear_call.arguments[0]
    assert isinstance(clear_constructor, ConstructorNode)
    assert clear_constructor.constructor_type.name == "RenderOutput::Clear"
    assert clear_constructor.arguments == []
    assert list(clear_constructor.named_arguments) == ["color", "depth"]
    assert isinstance(clear_constructor.named_arguments["color"], IdentifierNode)
    assert clear_constructor.named_arguments["color"].name == "color"


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


def test_global_io_location_attributes_parse():
    code = """
    shader Interface {
        float inputValue @input @location(3) @component(1) @flat @centroid;
        float outputValue @output @location(5) @index(1) @noperspective @sample @invariant @precise;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    input_value, output_value = ast.global_variables

    assert [attr.name for attr in input_value.attributes] == [
        "input",
        "location",
        "component",
        "flat",
        "centroid",
    ]
    assert isinstance(input_value.attributes[1].arguments[0], LiteralNode)
    assert input_value.attributes[1].arguments[0].value == 3
    assert input_value.attributes[2].arguments[0].value == 1
    assert [attr.name for attr in output_value.attributes] == [
        "output",
        "location",
        "index",
        "noperspective",
        "sample",
        "invariant",
        "precise",
    ]
    assert isinstance(output_value.attributes[1].arguments[0], LiteralNode)
    assert output_value.attributes[1].arguments[0].value == 5
    assert output_value.attributes[2].arguments[0].value == 1


def test_struct_member_layout_attributes_parse_with_post_attributes():
    code = """
    shader InterfaceStructLayouts {
        struct VSInput {
            layout(location = 0) vec3 position @flat;
            layout(location = 1, component = 2) vec2 uv @centroid;
        };
    }
    """

    ast = parse_code(tokenize_code(code))
    position, uv = ast.structs[0].members

    assert [attribute.name for attribute in position.attributes] == [
        "location",
        "flat",
    ]
    assert position.attributes[0].arguments[0].value == 0

    assert [attribute.name for attribute in uv.attributes] == [
        "location",
        "component",
        "centroid",
    ]
    assert [attribute.arguments[0].value for attribute in uv.attributes[:2]] == [1, 2]


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            "float value @input @location(0) @flat @noperspective;",
            "@flat and @noperspective",
        ),
        (
            "float value @input @location(0) @smooth @flat;",
            "@flat and @smooth",
        ),
        (
            "float value @input @location(0) @linear @nointerpolation;",
            "@linear and @nointerpolation",
        ),
        (
            "float value @input @location(0) @centroid @sample;",
            "@centroid and @sample",
        ),
        (
            "float value @input @location(0) @linear_centroid @sample;",
            "@linear_centroid and @sample",
        ),
    ],
)
def test_conflicting_interpolation_metadata_fails_validation(declaration, message):
    code = f"""
    shader ConflictingInterpolation {{
        {declaration}
    }}
    """

    with pytest.raises(ValueError, match=message):
        parse_code(tokenize_code(code))


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            "readonly image2D source @writeonly;",
            "readonly.*writeonly|writeonly.*readonly",
        ),
        (
            "writeonly image2D target @access(read);",
            "readonly.*writeonly|writeonly.*readonly",
        ),
    ],
)
def test_conflicting_resource_access_metadata_fails_validation(declaration, message):
    code = f"""
    shader ConflictingResourceAccess {{
        {declaration}
    }}
    """

    with pytest.raises(ValueError, match=message):
        parse_code(tokenize_code(code))


def test_conflicting_layout_attribute_values_fail_validation():
    code = """
    shader ConflictingLayoutValues {
        struct VSInput {
            layout(location = 0) vec3 position @location(1);
        };
    }
    """

    with pytest.raises(ValueError, match="Conflicting location metadata"):
        parse_code(tokenize_code(code))


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            "layout(binding = 0) image2D color @binding(1);",
            "Conflicting binding metadata",
        ),
        (
            "layout(set = 0) image2D color @set(1);",
            "Conflicting set metadata",
        ),
        (
            "image2D color @format(r32f) @format(rgba32f);",
            "Conflicting format metadata",
        ),
    ],
)
def test_conflicting_resource_layout_metadata_values_fail_validation(
    declaration, message
):
    code = f"""
    shader ConflictingResourceLayoutMetadata {{
        {declaration}
    }}
    """

    with pytest.raises(ValueError, match=message):
        parse_code(tokenize_code(code))


def test_duplicate_matching_resource_layout_metadata_values_are_allowed():
    code = """
    shader MatchingResourceLayoutMetadata {
        layout(set = 0, binding = 1) image2D color @set(0) @binding(1);
    }
    """

    ast = parse_code(tokenize_code(code))
    color = ast.global_variables[0]

    assert [attribute.name for attribute in color.attributes] == [
        "set",
        "binding",
        "set",
        "binding",
    ]
    assert [attribute.arguments[0].value for attribute in color.attributes] == [
        0,
        1,
        0,
        1,
    ]


def test_precise_variable_attribute_parse():
    code = """
    shader Precise {
        compute {
            void main() {
                float acc @precise = 0.0;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    statements = ast.stages[ShaderStage.COMPUTE].entry_point.body.statements

    assert isinstance(statements[0], VariableNode)
    assert [attr.name for attr in statements[0].attributes] == ["precise"]


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


def test_hlsl_tessellation_and_ray_stage_aliases_parse_to_canonical_stages():
    code = """
    shader AliasStages {
        hull HullMain {
            void main() { return; }
        }
        domain DomainMain {
            void main() { return; }
        }
        intersection IntersectionMain {
            void main() { return; }
        }
        closesthit ClosestHitMain {
            void main() { return; }
        }
        anyhit AnyHitMain {
            void main() { return; }
        }
        miss MissMain {
            void main() { return; }
        }
        callable CallableMain {
            void main() { return; }
        }
    }
    """

    ast = parse_code(tokenize_code(code))

    assert ast.stages[ShaderStage.TESSELLATION_CONTROL].entry_point.name == "HullMain"
    assert (
        ast.stages[ShaderStage.TESSELLATION_EVALUATION].entry_point.name == "DomainMain"
    )
    assert (
        ast.stages[ShaderStage.RAY_INTERSECTION].entry_point.name == "IntersectionMain"
    )
    assert ast.stages[ShaderStage.RAY_CLOSEST_HIT].entry_point.name == "ClosestHitMain"
    assert ast.stages[ShaderStage.RAY_ANY_HIT].entry_point.name == "AnyHitMain"
    assert ast.stages[ShaderStage.RAY_MISS].entry_point.name == "MissMain"
    assert ast.stages[ShaderStage.RAY_CALLABLE].entry_point.name == "CallableMain"


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

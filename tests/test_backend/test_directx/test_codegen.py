from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
import pytest
from typing import List


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = DirectxCrossGLCodeGen.HLSLToCrossGLConverter()
    return codegen.generate(ast_node)


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = HLSLLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST."""
    parser = HLSLParser(tokens)
    return parser.parse()


def test_struct_codegen():
    code = """
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;
        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color =  input.in_position;
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Struct parsing or code generation not implemented.")


def test_if_codegen():
    code = """
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;
        if (input.color.r > 0.5) {
            output.out_position = input.color;
        }
        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color =  input.in_position;
        if (input.in_position.r > 0.5) {
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("If statement parsing or code generation not implemented.")


def test_for_codegen():
    code = """
    #pragma exclude_renderers vulkan;
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;
        for (int i = 0; i < 10; i=i+1) {
            output.out_position = input.color;
        }
        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color =  input.in_position;
        for (int i = 0; i < 10; i=i+1) {
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")


def test_while_codegen():
    code = """
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position = input.position;
        int i = 0;
        while (i < 10) {
            output.out_position = input.color;
            i = i + 1;  // Increment the loop variable
        }
        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color = input.in_position;
        int i = 0;
        while (i < 10) {
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
            i = i + 1;  // Increment the loop variable
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("While loop parsing or code generation not implemented.")


def test_do_while_codegen():
    code = """
    struct VSInput {
        float4 position : POSITION;
        float4 color : TEXCOORD0;
    };
    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position = input.position;
        int i = 0;
        do {
            output.out_position = input.color;
            i = i + 1;  // Increment the loop variable
        } while (i < 10);
        return output;
    }
    struct PSInput {
        float4 in_position : TEXCOORD0;
    };
    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };
    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color = input.in_position;
        int i = 0;
        do {
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
            i = i + 1;  // Increment the loop variable
        } while (i < 10);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("While loop parsing or code generation not implemented.")


def test_else_codegen():
    code = """
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;
        if (input.color.r > 0.5) {
            output.out_position = input.color;
        }
        else {
            output.out_position = float4(0.0, 0.0, 0.0, 1.0);
        }
        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color =  input.in_position;
        if (input.in_position.r > 0.5) {
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
        }
        else {
            output.out_color = float4(0.0, 0.0, 0.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else statement parsing or code generation not implemented.")


def test_function_call_codegen():
    code = """
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    float4 Add(float4 a, float4 b) {
        return a + b;
    }
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;
        float4 result = Add(input.position, input.color);
        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color =  input.in_position;
        float4 result = Add(input.in_position, float4(1.0, 1.0, 1.0, 1.0));
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Function call parsing or code generation not implemented.")


def test_else_if_codegen():
    code = """
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;
        if (input.color.r > 0.5) {
            output.out_position = input.color;
        }
        else {
            output.out_position = float4(0.0, 0.0, 0.0, 1.0);
        }
        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        if (input.in_position.r > 0.5) {
            output.out_color = input.in_position;
        } else if (input.in_position.r == 0.5){
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
        } else {
            output.out_color = float4(0.0, 0.0, 0.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        # Print relevant tokens for debugging
        for i, token in enumerate(tokens):
            if "else" in str(token).lower() or "if" in str(token).lower():
                print(f"Token {i}: {token}")

        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else_if statement parsing or code generation not implemented.")


def test_assignment_ops_codegen():
    code = """
    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color = float4(0.0, 0.0, 0.0, 1.0);

        if (input.in_position.r > 0.5) {
            output.out_color += input.in_position;
        }

        if (input.in_position.r < 0.5) {
            output.out_color -= float4(0.1, 0.1, 0.1, 0.1);
        }

        if (input.in_position.g > 0.5) {
            output.out_color *= 2.0;
        }

        if (input.in_position.b > 0.5) {
            out_color /= 2.0;
        }

        // Testing SHIFT_LEFT (<<) operator on some condition
        if (input.in_position.r == 0.5) {
            uint redValue = asuint(output.out_color.r);
            output.redValue ^= 0x1;
            output.out_color.r = asfloat(redValue);

            output.redValue |= 0x2;
            // Applying shift left operation
            output.redValue << 1; // Shift left by 1
            output.redValue &= 0x3;
        }
        
        // Testing SHIFT_RIGHT (>>) operator on some condition
        if (input.in_position.r == 0.25) {
            uint redValue = asuint(output.out_color.r);
            output.redValue ^= 0x1;
            output.out_color.r = asfloat(redValue);

            output.redValue |= 0x2;
            // Applying shift left operation
            output.redValue >> 1; // Shift left by 1
            output.redValue &= 0x3;
        } 
        
        // Testing BITWISE_XOR (^) operator on some condition
        if (input.in_position.r == 0.5) {
            uint redValue = asuint(output.out_color.r);
            output.redValue ^ 0x1;  
            // BITWISE_XOR operation
            output.out_color.r = asfloat(redValue);
        }



        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("assignment ops parsing or code generation not implemented.")


def test_bitwise_ops_codgen():
    code = """
        PSOutput PSMain(PSInput input) {
            PSOutput output;
            output.out_color = float4(0.0, 0.0, 0.0, 1.0);
            uint val = 0x01;
            if (val | 0x02) {
                // Test case for bitwise OR
            }
            uint filterA = 0b0001; // First filter
            uint filterB = 0b1000; // Second filter

            // Merge both filters
            uint combinedFilter = filterA | filterB; // combinedFilter becomes 0b1001
            return output;
        }
        """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("bitwise_op parsing or codegen not implemented.")


def test_pragma_codegen():
    code = """
    #pragma exclude_renderers vulkan;
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;

        for (int i = 0; i < 10; i=i+1) {
            output.out_position = input.color;
        }

        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color =  input.in_position;

        for (int i = 0; i < 10; i=i+1) {
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
        }

        return output;
    }
    """

    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")
        pytest.fail("Include statement failed to parse or generate code.")


def test_include_codegen():
    code = """
    #include "common.hlsl"
    struct VSInput {
    float4 position : POSITION;
    float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position =  input.position;

        for (int i = 0; i < 10; i=i+1) {
            output.out_position = input.color;
        }

        return output;
    }

    struct PSInput {
        float4 in_position : TEXCOORD0;
    };

    struct PSOutput {
        float4 out_color : SV_TARGET0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color =  input.in_position;

        for (int i = 0; i < 10; i=i+1) {
            output.out_color = float4(1.0, 1.0, 1.0, 1.0);
        }

        return output;
    }
    """

    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")
        pytest.fail("Include statement failed to parse or generate code.")


def test_switch_case_codegen():
    code = """
    struct PSInput {
        float4 in_position : TEXCOORD0;
        int value : SV_InstanceID;
    };

    struct PSOutput {
        float4 out_color : SV_Target;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        switch (input.value) {
            case 1:
                output.out_color = float4(1.0, 0.0, 0.0, 1.0);
                break;
            case 2:
                output.out_color = float4(0.0, 1.0, 0.0, 1.0);
                break;
            default:
                output.out_color = float4(0.0, 0.0, 1.0, 1.0);
                break;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Switch-case parsing or code generation not implemented.")


def test_bitwise_and_ops_codgen():
    code = """
        PSOutput PSMain(PSInput input) {
            PSOutput output;
            output.out_color = float4(0.0, 0.0, 0.0, 1.0);
            uint val = 0x01;
            if (val & 0x02) {
                // Test case for bitwise AND
            }
            uint filterA = 0b0001; // First filter
            uint filterB = 0b1000; // Second filter

            // Merge both filters
            uint combinedFilter = filterA & filterB; // combinedFilter becomes 0b1001
            return output;
        }
        
        """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("bitwise_and_op codegen not implemented.")


def test_double_dtype_codegen():
    code = """
            PSOutput PSMain(PSInput input) {
                PSOutput output;
                output.out_color = float4(0.0, 0.0, 0.0, 1.0);
                double value1 = 3.14159; // First double value
                double value2 = 2.71828; // Second double value
                double result = value1 + value2; // Adding two doubles
                if (result > 6.0) {
                    output.out_color = float4(1.0, 0.0, 0.0, 1.0); // Set color to red
                } else {
                    output.out_color = float4(0.0, 1.0, 0.0, 1.0); // Set color to green
                }
                return output;
            }
        """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("double dtype parsing or code generation not implemented.")


def test_half_dtype_codegen():
    code = """
            PSOutput PSMain(PSInput input) {
                PSOutput output;
                output.out_color = float4(0.0, 0.0, 0.0, 1.0);
                half value1 = 3.14159; // First half value
                half value2 = 2.71828; // Second half value
                half result = value1 + value2; // Adding them
                if (result > 6.0) {
                    output.out_color = float4(1.0, 0.0, 0.0, 1.0); // Set color to red
                } else {
                    output.out_color = float4(0.0, 1.0, 0.0, 1.0); // Set color to green
                }
                return output;
            }
        """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("half dtype parsing or code generation not implemented.")


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


if __name__ == "__main__":
    pytest.main()

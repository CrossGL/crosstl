import pytest
from typing import List
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = HLSLParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = HLSLLexer(code)
    return lexer.tokenize()


def test_struct_parsing():
    code = """
    struct VSInput {
        float4 position : SV_Position;
        float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_parsing():
    code = """
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position = input.position;
        if (input.color.r > 0.5) {
            output.out_position = input.color;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("if parsing not implemented.")


def test_for_parsing():
    code = """
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        for (int i = 0; i < 10; i=i+1) {
            output.out_position = input.position;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("for parsing not implemented.")


def test_while_parsing():
    code = """
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        int i = 0;
        while (i < 10) {
            output.out_position = input.position;
            i = i + 1;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("while parsing not implemented")


def test_do_while_parsing():
    code = """
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        int i = 0;
        do {
            output.out_position = input.position;
            i = i + 1;
        } while (i < 10);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("do while parsing not implemented")


def test_else_parsing():
    code = """
    PSOutput PSMain(PSInput input) {
        PSOutput output;
        if (input.in_position.r > 0.5) {
            output.out_color = input.in_position;
        } else {
            output.out_color = float4(0.0, 0.0, 0.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("else parsing not implemented.")


def test_function_call_parsing():
    code = """
    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color = saturate(input.in_position);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("function call parsing not implemented.")


def test_else_if_parsing():
    code = """
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("else_if parsing not implemented.")


def test_assignment_ops_parsing():
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
            output.redValue ^ 0x1;  // BITWISE_XOR operation
            output.out_color.r = asfloat(redValue);
        }



        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("assign_op parsing not implemented.")


def test_bitwise_ops_parsing():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("bitwise_op parsing not implemented.")


def test_logical_or_ops_parsing():
    code = """
            PSOutput PSMain(PSInput input) {
                PSOutput output;
                output.out_color = float4(0.0, 0.0, 0.0, 1.0);
                // Test case for logical OR
                bool condition1 = true; // First condition
                bool condition2 = false; // Second condition
                if (condition1 || condition2) {
                    // If one of the condition is true
                    output.out_color = float4(1.0, 0.0, 0.0, 1.0); // Set color to red
                } else {
                    // If both of the conditions are false
                    output.out_color = float4(0.0, 1.0, 0.0, 1.0); // Set color to green
                }
                return output;
            }
        """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("logical_or_ops not implemented.")


def test_logical_and_ops_parsing():
    code = """
            PSOutput PSMain(PSInput input) {
                PSOutput output;
                output.out_color = float4(0.0, 0.0, 0.0, 1.0);
                // Test case for logical AND
                bool condition1 = true; // First condition
                bool condition2 = false; // Second condition
                if (condition1 && condition2) {
                    // both the condition is true
                    output.out_color = float4(1.0, 0.0, 0.0, 1.0); // Set color to red
                } else {
                    // any one of the condition is false
                    output.out_color = float4(0.0, 1.0, 0.0, 1.0); // Set color to green
                }
                return output;
            }
        """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("logical_and_ops not implemented.")


def test_switch_case_parsing():
    code = """
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("switch-case parsing not implemented.")


def test_double_dtype_parsing():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("double dtype not implemented.")


def test_mod_parsing():
    code = """
    void main() {
        int a = 10 % 3;  // Basic modulus
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


def test_double_dtype_parsing():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("half dtype not implemented.")


if __name__ == "__main__":
    pytest.main()

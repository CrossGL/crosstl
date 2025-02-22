import pytest
from typing import List
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = HLSLLexer(code)
    return lexer.tokenize()


def test_struct_tokenization():
    code = """
    struct VSInput {
        float4 position : SV_position;
        float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("struct tokenization not implemented.")


def test_if_tokenization():
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("if tokenization not implemented.")


def test_for_tokenization():
    code = """
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        for (int i = 0; i < 10; i++) {
            output.out_position += input.position;
        }
        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("for tokenization not implemented.")


def test_else_tokenization():
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("else tokenization not implemented.")


def test_function_call_tokenization():
    code = """
    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color = saturate(input.in_position);
        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")


def test_else_if_tokenization():
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("else_if tokenization not implemented.")


def test_assignment_ops_tokenization():
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
            output.out_color /= 2.0;
        }

        // Testing SHIFT_LEFT (<<) operator on some condition
        if (input.in_position.r == 0.5) {
            uint redValue = asuint(output.out_color.r);
            output.redValue ^= 0x1;
            output.out_color.r = asfloat(redValue);
            output.redValue |= 0x2;

            // Applying shift left operation
            output.redValue << 1; // Shift left by 1
            redValue |= 0x2;

            redValue &= 0x3;
        }
        
        // Testing SHIFT RIGHT (>>) operator on some condition
        if (input.in_position.r == 0.25) {
            uint redValue = asuint(output.out_color.r);
            output.redValue ^= 0x1;
            output.out_color.r = asfloat(redValue);
            output.redValue |= 0x2;

            // Applying shift left operation
            output.redValue >> 1; // Shift left by 1
            redValue |= 0x2;

            redValue &= 0x3;
        }


        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("assign_op tokenization is not implemented.")


def test_bitwise_or_tokenization():
    code = """
        uint val = 0x01;
        val = val | 0x02;
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("bitwise_op tokenization is not implemented.")


def test_logical_or_tokenization():
    code = """
        bool val_0 = true;
        bool val_1 = val_0 || false;
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("logical_or tokenization is not implemented.")


def test_logical_and_tokenization():
    code = """
        bool val_0 = true;
        bool val_1 = val_0 && false;
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("logical_and tokenization is not implemented.")


def test_switch_case_tokenization():
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("switch-case tokenization not implemented.")


def test_bitwise_and_tokenization():
    code = """
        uint val = 0x01;
        val = val & 0x02;
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("bitwise_and_op tokenization is not implemented.")


def test_double_dtype_tokenization():
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("double dtype tokenization is not implemented.")


def test_mod_tokenization():
    code = """
        int a = 10 % 3;  // Basic modulus
    """
    tokens = tokenize_code(code)

    # Find the modulus operator in tokens
    has_mod = False
    for token in tokens:
        if token == ("MOD", "%"):
            has_mod = True
            break

    assert has_mod, "Modulus operator (%) not tokenized correctly"


def test_half_dtype_tokenization():
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("half dtype tokenization is not implemented.")


def test_bitwise_not_tokenization():
    code = """
        int a = ~5;  // Bitwise NOT
    """
    tokens = tokenize_code(code)

    has_not = False
    for token in tokens:
        if token == ("BITWISE_NOT", "~"):
            has_not = True
            break

    assert has_not, "Bitwise NOT operator (~) not tokenized correctly"


if __name__ == "__main__":
    pytest.main()

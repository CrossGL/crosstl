from crosstl.src.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.src.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.src.backend.DirectX.DirectxParser import HLSLParser
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
    return lexer.tokens


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
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else_if statement parsing or code generation not implemented.")


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

        if (input.in_position.r == 0.5) {
            uint redValue = asuint(output.out_color.r);
            output.redValue ^= 0x1;
            output.out_color.r = asfloat(redValue);

            outpu.redValue |= 0x2;
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


if __name__ == "__main__":
    pytest.main()

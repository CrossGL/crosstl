from crosstl.backend.slang import SlangCrossGLCodeGen
from crosstl.backend.slang import SlangLexer
from crosstl.backend.slang import SlangParser
import pytest
from typing import List


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = SlangCrossGLCodeGen.SlangToCrossGLConverter()
    return codegen.generate(ast_node)


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = SlangLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST."""
    parser = SlangParser(tokens)
    return parser.parse()


def test_struct_codegen():
    code = """
    struct AssembledVertex
    {
    float3	position : POSITION;
    };
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
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
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
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        for (int i = 0; i < 10; i=i+1) {
            output.out_position += assembledVertex.position;
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


def test_else_codegen():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
        }
        else {
            output.out_position = float3(0.0, 0.0, 0.0);
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
    float4 saturate(float4 color) {
        return color;
    }

    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        output.out_position = saturate(assembledVertex.color);
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


if __name__ == "__main__":
    pytest.main()

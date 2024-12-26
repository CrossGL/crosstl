import pytest
from typing import List
from crosstl.translator.lexer import Lexer

def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.tokens

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
        assert any(t[0] == "STRUCT" for t in tokens), "Missing 'STRUCT' token"
    except SyntaxError:
        pytest.fail("Struct tokenization not implemented.")

def test_if_statement_tokenization():
    code = """
    if (a > b) {
        return a;
    } else {
        return b;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "IF" for t in tokens), "Missing 'IF' token"
        assert any(t[0] == "ELSE" for t in tokens), "Missing 'ELSE' token"
    except SyntaxError:
        pytest.fail("if tokenization not implemented.")

def test_for_statement_tokenization():
    code = """
    for (int i = 0; i < 10; i = i + 1) {
        sum += i;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "FOR" for t in tokens), "Missing 'FOR' token"
    except SyntaxError:
        pytest.fail("for tokenization not implemented.")

def test_else_statement_tokenization():
    code = """
    if (a > b) {
        return a;
    } else {
        return 0;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "ELSE" for t in tokens), "Missing 'ELSE' token"
    except SyntaxError:
        pytest.fail("else tokenization not implemented.")

def test_else_if_statement_tokenization():
    code = """
    if (!a) {
        return b;
    } 
    if (!b) {
        return a;
    } else if (a < b) {
        return b;
    } else if (a > b) {
        return a;
    } else {
        return 0;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert tokens, "No tokens generated"
    except SyntaxError:
        pytest.fail("else if tokenization not implemented.")

def test_function_call_tokenization():
    code = """
shader main {
    
    // Perlin Noise Function
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
    
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            bloom = perlinNoise(input.color.xy);
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}
"""
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "SHADER" for t in tokens), "Missing 'SHADER' token"
        assert any(t[0] == "FRAGMENT" for t in tokens), "Missing 'FRAGMENT' token"
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")

def test_bitwise_operator_tokenization():
    code = """
    int a = 60; // 60 = 0011 1100
    int b = 13; // 13 = 0000 1101
    int c = 0;
    c = a & b; 
    c = a | b; 
    c = a ^ b; 
    c = ~a; 
    c = a << 2; 
    c = a >> 2; 
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "BITWISE_AND" for t in tokens), "Missing '&' token"
        assert any(t[0] == "BITWISE_OR" for t in tokens), "Missing '|' token"
        assert any(t[0] == "BITWISE_XOR" for t in tokens), "Missing '^' token"
        assert any(t[0] == "BITWISE_NOT" for t in tokens), "Missing '~' token"
    except SyntaxError:
        pytest.fail("Bitwise operator tokenization not implemented.")

def test_data_types_tokenization():
    code = """
    int a;
    uint b;
    float c;
    double d;
    bool e;
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "INT" for t in tokens), "Missing 'INT' token"
        assert any(t[0] == "UINT" for t in tokens), "Missing 'UINT' token"
        assert any(t[0] == "FLOAT" for t in tokens), "Missing 'FLOAT' token"
        assert any(t[0] == "DOUBLE" for t in tokens), "Missing 'DOUBLE' token"
        assert any(t[0] == "BOOL" for t in tokens), "Missing 'BOOL' token"
    except SyntaxError:
        pytest.fail("Data types tokenization not implemented.")

def test_operators_tokenization():
    code = """
    int a;
    a = 2 + 1;
    a = a - 2;
    a = a / 1;
    a = a * 2;
    a = a % 2;
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "PLUS" for t in tokens), "Missing '+' token"
        assert any(t[0] == "MINUS" for t in tokens), "Missing '-' token"
        assert any(t[0] == "DIVIDE" for t in tokens), "Missing '/' token"
        assert any(t[0] == "MULTIPLY" for t in tokens), "Missing '*' token"
        assert any(t[0] == "MOD" for t in tokens), "Missing '%' token"
    except SyntaxError:
        pytest.fail("Operators tokenization not implemented.")

def test_logical_operators_tokenization():
    code = """
    if (0.8 > 0.7 || 0.6 > 0.7) {    
        return 0;
    } else if(0.8 > 0.7 && 0.8> 0.7) {        
        return 1;  
    }
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "LOGICAL_OR" for t in tokens), "Missing '||' token"
        assert any(t[0] == "LOGICAL_AND" for t in tokens), "Missing '&&' token"
    except SyntaxError:
        pytest.fail("Logical Operators tokenization not implemented.")

def test_assignment_shift_operators():
    code = """
    a >>= 1;
    b <<= 1;
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "ASSIGN_SHIFT_RIGHT" for t in tokens), "Missing '>>=' token"
        assert any(t[0] == "ASSIGN_SHIFT_LEFT" for t in tokens), "Missing '<<=' token"
    except SyntaxError:
        pytest.fail("Shift operators tokenization failed.")

def test_assignment_operators_tokenization():
    code = """
    int a = 1;
    a += 1;
    a *= 2;
    a /= a;
    a -= -1;
    a %= 2;
    a &= 1;
    a |= 1;
    a ^= 1;
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "ASSIGN_ADD" for t in tokens), "Missing '+=' token"
        assert any(t[0] == "ASSIGN_MUL" for t in tokens), "Missing '*=' token"
        assert any(t[0] == "ASSIGN_DIV" for t in tokens), "Missing '/=' token"
        assert any(t[0] == "ASSIGN_SUB" for t in tokens), "Missing '-=' token"
        assert any(t[0] == "ASSIGN_MOD" for t in tokens), "Missing '%=' token"
        assert any(t[0] == "ASSIGN_AND" for t in tokens), "Missing '&=' token"
        assert any(t[0] == "ASSIGN_OR" for t in tokens), "Missing '|=' token"
        assert any(t[0] == "ASSIGN_XOR" for t in tokens), "Missing '^=' token"
    except SyntaxError:
        pytest.fail("Assignment operators tokenization not implemented.")

def test_const_tokenization():
    code = """
    const int a;
    """
    try:
        tokens = tokenize_code(code)
        assert any(t[0] == "CONST" for t in tokens), "Missing 'CONST' token"
    except SyntaxError:
        pytest.fail("Const keyword tokenization failed")

def test_illegal_character():
    code = "int a = 1 @#"
    with pytest.raises(SyntaxError):
        tokenize_code(code)

if __name__ == "__main__":
    pytest.main()
import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("STRUCT", r"struct"),
    ("CBUFFER", r"cbuffer"),
    ("TEXTURE2D", r"Texture2D"),
    ("SAMPLER_STATE", r"SamplerState"),
    ("FVECTOR", r"float[2-4]"),
    ("FLOAT", r"float"),
    ("INT", r"int"),
    ("UINT", r"uint"),
    ("BOOL", r"bool"),
    ("MATRIX", r"float[2-4]x[2-4]"),
    ("VOID", r"void"),
    ("RETURN", r"return"),
    ("IF", r"if"),
    ("ELSE", r"else"),
    ("FOR", r"for"),
    ("REGISTER", r"register"),
    ("SEMANTIC", r": [A-Z_][A-Z0-9_]*"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NUMBER", r"\d+(\.\d+)?"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("SEMICOLON", r";"),
    ("COMMA", r","),
    ("COLON", r":"),
    ("EQUALS", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("LESS_THAN", r"<"),
    ("GREATER_THAN", r">"),
    ("LESS_EQUAL", r"<="),
    ("GREATER_EQUAL", r">="),
    ("EQUAL", r"=="),
    ("NOT_EQUAL", r"!="),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("DOT", r"\."),
    ("WHITESPACE", r"\s+"),
]

KEYWORDS = {
    "struct": "STRUCT",
    "cbuffer": "CBUFFER",
    "Texture2D": "TEXTURE2D",
    "SamplerState": "SAMPLER_STATE",
    "float": "FLOAT",
    "float2": "FVECTOR",
    "float3": "FVECTOR",
    "float4": "FVECTOR",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "void": "VOID",
    "return": "RETURN",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "register": "REGISTER",
}


class HLSLLexer:
    def __init__(self, code):
        self.code = code
        self.tokens = []
        self.tokenize()

    def tokenize(self):
        pos = 0
        while pos < len(self.code):
            match = None
            for token_type, pattern in TOKENS:
                regex = re.compile(pattern)
                match = regex.match(self.code, pos)
                if match:
                    text = match.group(0)
                    if token_type == "IDENTIFIER" and text in KEYWORDS:
                        token_type = KEYWORDS[text]
                    if token_type not in [
                        "WHITESPACE",
                        "COMMENT_SINGLE",
                        "COMMENT_MULTI",
                    ]:
                        token = (token_type, text)
                        self.tokens.append(token)
                    pos = match.end(0)
                    break
            if not match:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )

        self.tokens.append(("EOF", ""))


if __name__ == "__main__":
    hlsl_code = """
struct VS_INPUT {
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
};

struct PS_INPUT {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
};

Texture2D mainTexture : register(t0);
SamplerState mainSampler : register(s0);

float4 BlurPixel(float2 texCoord, float2 texelSize, float blurAmount)
{
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    
    for (int x = -3; x <= 3; ++x)
    {
        for (int y = -3; y <= 3; ++y)
        {
            float2 offset = float2(float(x), float(y)) * texelSize * blurAmount;
            color += mainTexture.Sample(mainSampler, texCoord + offset);
        }
    }
    
    return color / 49.0;
}

PS_INPUT VSMain(VS_INPUT input)
{
    PS_INPUT output;
    output.position = float4(input.position, 1.0);
    output.texCoord = input.texCoord;
    return output;
}

float4 PSMain(PS_INPUT input) : SV_TARGET
{
    float2 texelSize = 1.0 / float2(1920, 1080); // Assume 1920x1080 resolution
    float blurAmount = 1.0;
    return BlurPixel(input.texCoord, texelSize, blurAmount);
}
"""

    lexer = HLSLLexer(hlsl_code)
    for token in lexer.tokens:
        print(token)

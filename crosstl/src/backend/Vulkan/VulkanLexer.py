import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("WHITESPACE", r"\s+"),
    ("SEMANTIC", r":\w+"),  
    ("PRE_INCREMENT", r"\+\+(?=\w)"),
    ("PRE_DECREMENT", r"--(?=\w)"),
    ("POST_INCREMENT", r"(?<=\w)\+\+"),
    ("POST_DECREMENT", r"(?<=\w)--"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),  
    ("NUMBER", r"\d+(\.\d*)?|\.\d+"),
    ("SEMICOLON", r";"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("PLUS_EQUALS", r"\+="),  
    ("MINUS_EQUALS", r"-="),
    ("MULTIPLY_EQUALS", r"\*="),
    ("DIVIDE_EQUALS", r"/="),
    ("EQUALS", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("MODULUS", r"%"),
    ("LESS_EQUAL", r"<="),  
    ("GREATER_EQUAL", r">="),
    ("NOT_EQUAL", r"!="),
    ("LESS_THAN", r"<"),
    ("GREATER_THAN", r">"),
    ("SHIFT_LEFT", r"<<"),
    ("SHIFT_RIGHT", r">>"),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("BINARY_AND", r"&"),
    ("BINARY_OR", r"\|"),
    ("BINARY_XOR", r"\^"),
    ("BINARY_NOT", r"~"),
    ("QUESTION", r"\?"),
    ("COLON", r":"),
]

KEYWORDS = {
    "struct": "STRUCT",
    "layout": "LAYOUT",
    "uniform": "UNIFORM",
    "sampler2D": "SAMPLER2D",
    "samplerCube": "SAMPLERCUBE",
    "vec2": "VEC2",
    "vec3": "VEC3",
    "vec4": "VEC4",
    "ivec2": "IVEC2",
    "ivec3": "IVEC3",
    "ivec4": "IVEC4",
    "uvec2": "UVEC2",
    "uvec3": "UVEC3",
    "uvec4": "UVEC4",
    "bvec2": "BVEC2",
    "bvec3": "BVEC3",
    "bvec4": "BVEC4",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "float": "FLOAT",
    "double": "DOUBLE",
    "void": "VOID",
    "return": "RETURN",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "do": "DO",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
    "continue": "CONTINUE",
    "discard": "DISCARD",
    "in": "IN",
    "out": "OUT",
    "inout": "INOUT",
    "attribute": "ATTRIBUTE",
    "varying": "VARYING",
    "const": "CONST",
    "precision": "PRECISION",
    "highp": "HIGHP",
    "mediump": "MEDIUMP",
    "lowp": "LOWP",
    "subpassInput": "SUBPASSINPUT",
    "subpassInputMS": "SUBPASSINPUTMS",
    "sampler2DArray": "SAMPLER2DARRAY",
    "sampler2DMS": "SAMPLER2DMS",
    "sampler2DMSArray": "SAMPLER2DMSARRAY",
    "sampler3D": "SAMPLER3D",
    "samplerCubeArray": "SAMPLERCUBEARRAY",
    "image2D": "IMAGE2D",
    "image3D": "IMAGE3D",
    "imageCube": "IMAGECUBE",
    "imageBuffer": "IMAGEBUFFER",
    "image2DArray": "IMAGE2DARRAY",
    "imageCubeArray": "IMAGECUBEARRAY",
    "image1D": "IMAGE1D",
    "image1DArray": "IMAGE1DARRAY",
    "image2DMS": "IMAGE2DMS",
    "image2DMSArray": "IMAGE2DMSARRAY",
    "atomic_uint": "ATOMICUINT",
    "mat2": "MAT2",
    "mat3": "MAT3",
    "mat4": "MAT4",
}

VALID_DATA_TYPES = ["int", "float", "double", "vec2", "vec3", "vec4", "mat2", "mat3", "mat4", "uint", "bool", "void"]

class VulkanLexer:
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
                    if token_type == "VERSION":
                        self.tokens.append((token_type, text))
                    elif token_type == "VERSION_NUMBER":
                        self.tokens.append((token_type, text))
                    elif token_type == "CORE":
                        self.tokens.append((token_type, text))
                    elif token_type != "WHITESPACE":  # Ignore whitespace tokens
                        token = (token_type, text)
                        self.tokens.append(token)
                    pos = match.end(0)
                    break
            if not match:
                unmatched_char = self.code[pos]
                highlighted_code = (
                    self.code[:pos] + "[" + self.code[pos] + "]" + self.code[pos + 1 :]
                )
                raise SyntaxError(
                    f"Illegal character '{unmatched_char}' at position {pos}\n{highlighted_code}"
                )

        self.tokens.append(("EOF", None))
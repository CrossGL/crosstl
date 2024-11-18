import re

# Define the tokens for Mojo syntax
TOKENS = [
    ("COMMENT_SINGLE", r"#.*"),
    ("COMMENT_MULTI", r'"""[\s\S]*?"""'),
    ("STRUCT", r"\bstruct\b"),
    ("LET", r"\blet\b"),
    ("VAR", r"\bvar\b"),
    ("FN", r"\bfn\b"),
    ("RETURN", r"\breturn\b"),
    ("IF", r"\bif\b"),
    ("ELSE", r"\belse\b"),
    ("FOR", r"\bfor\b"),
    ("WHILE", r"\bwhile\b"),
    ("IMPORT", r"\bimport\b"),
    ("DEF", r"\bdef\b"),
    ("INT", r"\bInt\b"),
    ("FLOAT", r"\bFloat\b"),
    ("BOOL", r"\bBool\b"),
    ("STRING", r"\bString\b"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NUMBER", r"\d+(\.\d+)?"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("SEMICOLON", r";"),
    ("STRING_LITERAL", r'"[^"]*"'),
    ("COMMA", r","),
    ("COLON", r":"),
    ("LESS_EQUAL", r"<="),
    ("GREATER_EQUAL", r">="),
    ("LESS_THAN", r"<"),
    ("GREATER_THAN", r">"),
    ("EQUAL", r"=="),
    ("NOT_EQUAL", r"!="),
    ("PLUS_EQUALS", r"\+="),
    ("MINUS_EQUALS", r"-="),
    ("MULTIPLY_EQUALS", r"\*="),
    ("DIVIDE_EQUALS", r"/="),
    ("POINTER", r"->"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("LOGICAL_NAND", r"!&&"),  
    ("LOGICAL_XOR", r"\^"),
    ("LOGICAL_NOR", r"\bnor\b"),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("NOT", r"!"),
    ("DOT", r"\."),
    ("EQUALS", r"="),
    ("WHITESPACE", r"\s+"),
]

# Define keywords specific to mojo
KEYWORDS = {
    "struct": "STRUCT",
    "let": "LET",
    "var": "VAR",
    "fn": "FN",
    "return": "RETURN",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "in": "IN",
    "while": "WHILE",
    "import": "IMPORT",
    "def": "DEF",
    "Int": "INT",
    "Float": "FLOAT",
    "Bool": "BOOL",
    "String": "STRING",
    "print": "PRINT",
}


class MojoLexer:
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


# Temp test sorry aha
code = """
fn test_logical_operations() {
    let a: Bool = true;
    let b: Bool = false;
    let c: Bool = true;
    let d: Bool = false;
    let e: Bool = true;
    let f: Bool = false;
    let g: Bool = true;

    let result_and = a && b;
    print("a AND b = ", result_and);

    
    let result_or = a || b;   
    print("a OR b = ", result_or);

    
    let result_xor = a != b; 
    print("a XOR b = ", result_xor);

    
    let result_not = !a;     
    print("NOT a = ", result_not);

    
    let result_nand = !(a && b);  
    print("a NAND b = ", result_nand);

}


test_logical_operations();
"""

lexer = MojoLexer(code)
# for token in lexer.tokens:
#       print(token)

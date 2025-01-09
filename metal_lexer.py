import re


class MetalLexer:
    def __init__(self, code, tokens):
        self.code = code
        self.pos = 0
        self.tokens = tokens

        # Combine all patterns into a single regex
        self.master_regex = re.compile(
            "|".join(
                f"(?P<{token_type}>{pattern})" for token_type, pattern in self.tokens
            )
        )

    def tokenize(self):
        while self.pos < len(self.code):
            match = self.master_regex.match(self.code, self.pos)
            if not match:
                raise ValueError(
                    f"Unexpected character at position {self.pos}: {self.code[self.pos]}"
                )

            token_type = match.lastgroup
            token_value = match.group(token_type)

            yield token_type, token_value

            # Update position
            self.pos = match.end()


# Example usage
TOKENS = [
    ("NUMBER", r"\d+"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("WHITESPACE", r"\s+"),  # We might want to skip this during tokenization
]

code = "12 + 34 - 56"
lexer = MetalLexer(code, TOKENS)

tokens = list(lexer.tokenize())
for token_type, token_value in tokens:
    if token_type != "WHITESPACE":  # Optionally skip whitespace tokens
        print(f"{token_type}: {token_value}")

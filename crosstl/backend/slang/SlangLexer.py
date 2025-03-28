"""
Slang lexer implementation
"""


class SlangLexer:
    """Lexer for Slang shading language."""

    def __init__(self, code):
        """Initialize lexer with Slang code.

        Args:
            code (str): Slang source code to tokenize
        """
        self.code = code
        self.keywords = {
            "float": "FLOAT_KW",
            "float4": "FLOAT4",
            "float2": "FLOAT2",
            "return": "RETURN",
        }

    def tokenize(self):
        """Tokenize the Slang code.

        Returns:
            list: List of (token_type, token_value) tuples
        """
        # Very basic lexer implementation for testing
        tokens = []
        # Split the code into lines
        lines = self.code.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split line into words and symbols
            current_word = ""
            i = 0
            while i < len(line):
                char = line[i]

                # Handle whitespace
                if char.isspace():
                    if current_word:
                        self._add_token(tokens, current_word)
                        current_word = ""
                    i += 1
                    continue

                # Handle symbols
                if char in "(){};=,:":
                    if current_word:
                        self._add_token(tokens, current_word)
                        current_word = ""

                    # Add the symbol token
                    if char == "{":
                        tokens.append(("LBRACE", char))
                    elif char == "}":
                        tokens.append(("RBRACE", char))
                    elif char == "(":
                        tokens.append(("LPAREN", char))
                    elif char == ")":
                        tokens.append(("RPAREN", char))
                    elif char == ";":
                        tokens.append(("SEMICOLON", char))
                    elif char == "=":
                        tokens.append(("EQUALS", char))
                    elif char == ",":
                        tokens.append(("COMMA", char))
                    elif char == ":":
                        tokens.append(("COLON", char))
                    i += 1
                    continue

                # Handle numbers
                if char.isdigit() or (
                    char == "." and i + 1 < len(line) and line[i + 1].isdigit()
                ):
                    if (
                        current_word
                        and not current_word[-1].isdigit()
                        and current_word[-1] != "."
                    ):
                        self._add_token(tokens, current_word)
                        current_word = ""

                    # Add digit to current word
                    current_word += char
                    i += 1
                    continue

                # Add character to current word
                current_word += char
                i += 1

            # Add any remaining word
            if current_word:
                self._add_token(tokens, current_word)

        return tokens

    def _add_token(self, tokens, word):
        """Add a token to the list based on the word type.

        Args:
            tokens (list): List to add the token to
            word (str): Word to tokenize
        """
        # Check if it's a keyword
        if word in self.keywords:
            tokens.append((self.keywords[word], word))
        # Check if it's a number
        elif word.replace(".", "", 1).isdigit():
            tokens.append(("NUMBER", word))
        # Special identifiers
        elif word == "SV_TARGET":
            tokens.append(("SV_TARGET", word))
        elif word == "TEXCOORD":
            tokens.append(("TEXCOORD", word))
        # Regular identifier
        else:
            tokens.append(("IDENTIFIER", word))

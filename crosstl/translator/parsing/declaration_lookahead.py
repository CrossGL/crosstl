"""Declaration lookahead in the CrossGL parser: generic, function and type token checks, type skipping, balanced token text, unexpected-token refusals and static assertions."""

from ..refusals import CrossGLSyntaxError
from ..static_assertion_syntax import (
    parse_static_assertion,
    starts_static_assertion,
)


class ParserDeclarationLookaheadMixin:
    """Declaration lookahead members of ``Parser``."""

    def is_generic_declaration(self):
        """Return whether the current token begins a legacy generic wrapper."""
        return (
            self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "generic"
        )

    def current_token_starts_bare_function_name(self):
        """Return whether the current token is a ``fn name`` style function name."""
        if not (
            self.current_token[0] == "KERNEL"
            or self.current_token_is_binding_identifier()
        ):
            return False

        offset = 1
        if self.peek(offset)[0] == "LESS_THAN":
            depth = 1
            offset += 1
            while depth > 0 and self.peek(offset)[0] != "EOF":
                if self.peek(offset)[0] == "LESS_THAN":
                    depth += 1
                elif self.peek(offset)[0] == "GREATER_THAN":
                    depth -= 1
                elif self.peek(offset)[0] == "BITWISE_SHIFT_RIGHT":
                    depth -= 2
                offset += 1

        return self.peek(offset)[0] == "LPAREN"

    def is_arrow_token(self):
        """Return whether the current token represents ``->``."""
        return self.current_token[0] == "ARROW" or (
            self.current_token[0] == "MINUS" and self.peek()[0] == "GREATER_THAN"
        )

    def eat_arrow(self):
        """Consume ``->`` whether the lexer emitted it as one token or two."""
        if self.current_token[0] == "ARROW":
            self.eat("ARROW")
        else:
            self.eat("MINUS")
            self.eat("GREATER_THAN")

    def is_function_declaration(self):
        """Return whether the current token sequence looks like a function."""
        saved_pos = self.pos
        saved_token = self.current_token
        token_mark = self.mark_token_edits()

        try:
            while (
                self.current_token[0]
                in {
                    "ASYNC",
                    "UNSAFE",
                    "GLOBAL",
                    "KERNEL",
                    "AT",
                    "ATTRIBUTE",
                }
                or self.current_token_starts_square_attribute()
                or self.current_token_is_function_precise_qualifier()
            ):
                if self.current_token[0] in ["AT", "ATTRIBUTE"]:
                    self.parse_attributes()
                elif self.current_token_starts_square_attribute():
                    self.parse_square_attributes()
                else:
                    self.eat(self.current_token[0])

            if self.current_token[0] == "FUNCTION":
                self.eat("FUNCTION")
                if self.current_token_starts_bare_function_name():
                    return True

            self.parse_parameter_qualifiers()
            if self.is_type_token():
                self.advance_over_type()
                self.advance_over_pointer_suffix()
                if (
                    self.current_token[0] == "KERNEL"
                    or self.current_token_is_binding_identifier()
                ):
                    self.parse_binding_identifier()
                    # Skip generic parameters if present
                    if self.current_token[0] == "LESS_THAN":
                        depth = 1
                        self.eat("LESS_THAN")
                        while depth > 0 and self.current_token[0] != "EOF":
                            if self.current_token[0] == "LESS_THAN":
                                depth += 1
                                self.eat("LESS_THAN")
                                continue
                            if self.current_token_is_type_greater_than():
                                depth -= 1
                                self.eat_type_greater_than()
                                continue
                            self.eat(self.current_token[0])

                    if self.current_token[0] == "LPAREN":
                        return True
        except:
            pass
        finally:
            self.undo_token_edits(token_mark)
            self.pos = saved_pos
            self.current_token = saved_token

        return False

    def current_token_is_function_precise_qualifier(self):
        """Return whether ``precise`` qualifies the result of a declared function.

        ``precise float f(...)`` requires the computation of the function's
        results without value-changing optimizations such as contraction, as
        ``precise`` does for a variable. The qualifier is kept on the function
        instead of being skipped as an unknown token.
        """
        token_type, token_value = self.current_token
        return (
            token_type == "IDENTIFIER"
            and token_value == "precise"
            and self.peek()[0] not in {"LPAREN", "EQUALS", "SEMICOLON", "COMMA"}
        )

    def is_type_token(self):
        """Return whether the current token can start a type expression."""
        return self.current_token[0] in [
            "BOOL",
            "I8",
            "I16",
            "I32",
            "I64",
            "U8",
            "U16",
            "U32",
            "U64",
            "F16",
            "BFLOAT16",
            "F32",
            "F64",
            "INT",
            "UINT",
            "FLOAT",
            "DOUBLE",
            "HALF",
            "CHAR",
            "STRING",
            "VOID",
            "TEXTURE1D",
            "TEXTURE2D",
            "TEXTURE3D",
            "TEXTURECUBE",
            "TEXTURE2DARRAY",
            "VEC2",
            "VEC3",
            "VEC4",
            "IVEC2",
            "IVEC3",
            "IVEC4",
            "UVEC2",
            "UVEC3",
            "UVEC4",
            "DVEC2",
            "DVEC3",
            "DVEC4",
            "BVEC2",
            "BVEC3",
            "BVEC4",
            "MAT2",
            "MAT3",
            "MAT4",
            "MAT2X2",
            "MAT2X3",
            "MAT2X4",
            "MAT3X2",
            "MAT3X3",
            "MAT3X4",
            "MAT4X2",
            "MAT4X3",
            "MAT4X4",
            "DMAT2",
            "DMAT3",
            "DMAT4",
            "DMAT2X2",
            "DMAT2X3",
            "DMAT2X4",
            "DMAT3X2",
            "DMAT3X3",
            "DMAT3X4",
            "DMAT4X2",
            "DMAT4X3",
            "DMAT4X4",
            "SAMPLER",
            "SAMPLER1D",
            "SAMPLER1DARRAY",
            "SAMPLER2D",
            "SAMPLER3D",
            "SAMPLERCUBE",
            "SAMPLER2DARRAY",
            "SAMPLER2DSHADOW",
            "SAMPLER2DARRAYSHADOW",
            "SAMPLERCUBESHADOW",
            "SAMPLERCUBEARRAY",
            "SAMPLERCUBEARRAYSHADOW",
            "SAMPLER2DMS",
            "SAMPLER2DMSARRAY",
            "ISAMPLER1D",
            "ISAMPLER1DARRAY",
            "ISAMPLER2D",
            "ISAMPLER2DARRAY",
            "ISAMPLER3D",
            "ISAMPLERCUBE",
            "ISAMPLERCUBEARRAY",
            "ISAMPLER2DMS",
            "ISAMPLER2DMSARRAY",
            "USAMPLER1D",
            "USAMPLER1DARRAY",
            "USAMPLER2D",
            "USAMPLER2DARRAY",
            "USAMPLER3D",
            "USAMPLERCUBE",
            "USAMPLERCUBEARRAY",
            "USAMPLER2DMS",
            "USAMPLER2DMSARRAY",
            "IIMAGE1D",
            "IIMAGE1DARRAY",
            "IIMAGE2D",
            "IIMAGE3D",
            "IIMAGECUBE",
            "IIMAGECUBEARRAY",
            "IIMAGE2DARRAY",
            "IIMAGE2DMS",
            "IIMAGE2DMSARRAY",
            "UIMAGE1D",
            "UIMAGE1DARRAY",
            "UIMAGE2D",
            "UIMAGE3D",
            "UIMAGECUBE",
            "UIMAGECUBEARRAY",
            "UIMAGE2DARRAY",
            "UIMAGE2DMS",
            "UIMAGE2DMSARRAY",
            "IMAGE1D",
            "IMAGE1DARRAY",
            "IMAGE2D",
            "IMAGE3D",
            "IMAGECUBE",
            "IMAGECUBEARRAY",
            "IMAGE2DARRAY",
            "IMAGE2DMS",
            "IMAGE2DMSARRAY",
            "IDENTIFIER",
        ]

    def advance_over_type(self):
        """Advance over a type expression during lookahead checks."""
        if self.is_type_token():
            self.eat(self.current_token[0])

            while self.current_token[0] in {"DOT", "DOUBLE_COLON"}:
                self.eat(self.current_token[0])
                if self.current_token_is_binding_identifier():
                    self.eat(self.current_token[0])
                else:
                    break

            if self.current_token[0] == "LESS_THAN":
                depth = 1
                self.eat("LESS_THAN")
                while depth > 0 and self.current_token[0] != "EOF":
                    if self.current_token[0] == "LPAREN":
                        # ``Loc<N, (N > 2)>``: a parenthesized value argument's ``<`` and ``>`` are comparisons.
                        self.parse_balanced_parenthesized_token_text()
                        continue
                    if self.current_token[0] == "LESS_THAN":
                        depth += 1
                        self.eat("LESS_THAN")
                        continue
                    if self.current_token_is_type_greater_than():
                        depth -= 1
                        self.eat_type_greater_than()
                        continue
                    self.eat(self.current_token[0])

            while self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                bracket_depth = 1
                while bracket_depth > 0 and self.current_token[0] != "EOF":
                    if self.current_token[0] == "LBRACKET":
                        bracket_depth += 1
                    elif self.current_token[0] == "RBRACKET":
                        bracket_depth -= 1
                    self.eat(self.current_token[0])

            while self.current_token[0] == "LPAREN":
                self.parse_balanced_parenthesized_token_text()

    def parse_balanced_parenthesized_token_text(self):
        """Consume a parenthesized token payload and return compact source text."""
        self.eat("LPAREN")
        payload_tokens = []
        depth = 1

        while depth and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token

            if token_type == "LPAREN":
                depth += 1
                payload_tokens.append((token_type, token_value))
                self.eat("LPAREN")
                continue

            if token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
                payload_tokens.append((token_type, token_value))
                self.eat("RPAREN")
                continue

            payload_tokens.append((token_type, token_value))
            self.eat(token_type)

        return self.format_raw_lambda_tokens(payload_tokens)

    def reject_unexpected_token(self, construct):
        """Raise ``SyntaxError`` for a token that starts nothing ``construct`` can hold.

        Skipping the token would silently drop the declaration or statement it belongs to.
        """
        token_type, token_value = self.current_token
        if token_type == "EOF":
            raise CrossGLSyntaxError(f"Unexpected end of input in {construct}")
        raise CrossGLSyntaxError(
            f"Unexpected {token_type} {token_value!r} in {construct}"
        )

    def parse_scope_static_assertion(self):
        """Parse a ``static_assert`` outside a block into ``self.static_assertions``; False if none starts here."""
        if not starts_static_assertion(self):
            return False
        self.static_assertions.append(parse_static_assertion(self))
        return True

    def reject_end_of_input(self, construct):
        """Raise ``SyntaxError`` if the input ends inside ``construct`` before its closing brace.

        A brace-delimited loop whose member parser returns ``None`` at ``EOF`` would otherwise spin forever on
        truncated input.
        """
        if self.current_token[0] == "EOF":
            raise CrossGLSyntaxError(
                f"Unterminated {construct}: expected '}}' before end of input"
            )

    def skip_balanced_declaration(self):
        """Recover past one semicolon or braced declaration without leaking tokens."""
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]

            if token_type == "RBRACE" and brace_depth == 0:
                break

            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                self.eat("SEMICOLON")
                break

            if token_type == "LBRACE":
                brace_depth += 1
                self.eat("LBRACE")
                continue

            if token_type == "RBRACE":
                self.eat("RBRACE")
                brace_depth -= 1
                if brace_depth == 0 and paren_depth == 0 and bracket_depth == 0:
                    break
                continue

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1

            self.eat(token_type)

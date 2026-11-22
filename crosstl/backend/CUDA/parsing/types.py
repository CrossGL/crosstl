"""Types in the CUDA parser: type names, decltype and elaborated types."""

from crosstl.translator.refusals import CudaSyntaxError


class CudaTypeMixin:
    """Type members of ``CudaParser``."""

    def parse_type(self):
        type_parts = []
        saw_integral_sign = False

        while self.current_token[0] in self.TYPE_QUALIFIER_TOKENS:
            if self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif self.current_token[0] in self.ELABORATED_TYPE_TOKENS:
            type_parts.append(self.parse_elaborated_type_name())
        elif self.is_global_qualified_name_start_at_index(self.current_index):
            type_parts.append(self.parse_type_name())
        elif self.current_token[0] in self.TYPE_TOKENS:
            type_parts.append(self.parse_type_name())

        self.parse_postfix_type_qualifiers(type_parts)
        while self.current_token[0] == "MULTIPLY":
            type_parts.append("*")
            self.eat("MULTIPLY")
            self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] in self.TYPE_REFERENCE_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] == "NUMBER":
                type_parts.append(f"[{self.current_token[1]}]")
                self.eat("NUMBER")
            else:
                type_parts.append("[]")
            self.eat("RBRACKET")

        return " ".join(type_parts)

    def parse_type_without_array_suffix(self):
        type_parts = []
        saw_integral_sign = False

        while self.current_token[0] in self.TYPE_QUALIFIER_TOKENS:
            if self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif self.current_token[0] in self.ELABORATED_TYPE_TOKENS:
            type_parts.append(self.parse_elaborated_type_name())
        elif self.is_global_qualified_name_start_at_index(self.current_index):
            type_parts.append(self.parse_type_name())
        elif self.current_token[0] in self.TYPE_TOKENS:
            type_parts.append(self.parse_type_name())

        self.parse_postfix_type_qualifiers(type_parts)
        while self.current_token[0] == "MULTIPLY":
            type_parts.append("*")
            self.eat("MULTIPLY")
            self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] in self.TYPE_REFERENCE_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(type_parts)

        return " ".join(type_parts)

    def is_implicit_int_current_type(self, saw_integral_sign):
        return self.is_implicit_int_type_at_index(self.current_index, saw_integral_sign)

    def is_decltype_type_start(self):
        return self.is_decltype_type_start_at_index(self.current_index)

    def is_decltype_type_start_at_index(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "IDENTIFIER"
            and self.tokens[index][1] == "decltype"
            and self.tokens[index + 1][0] == "LPAREN"
        )

    def skip_decltype_type_at_index(self, index):
        if not self.is_decltype_type_start_at_index(index):
            return None

        index += 1
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "EOF":
                return None
            index += 1

        return None

    def parse_decltype_type_name(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        parts = []
        depth = 1

        while depth > 0:
            token_type, token_value = self.current_token
            if token_type == "EOF":
                raise CudaSyntaxError("Unterminated decltype type")
            if token_type == "LPAREN":
                depth += 1
                parts.append(token_value)
                self.eat("LPAREN")
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
                parts.append(token_value)
                self.eat("RPAREN")
            else:
                parts.append(token_value)
                self.eat(token_type)

        return f"decltype({self.format_template_parts(parts)})"

    def parse_postfix_type_qualifiers(self, type_parts):
        while self.current_token[0] in self.POSTFIX_TYPE_QUALIFIER_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

    def parse_type_name(self):
        token_type = self.current_token[0]
        if token_type == "SCOPE":
            self.eat("SCOPE")
            type_name = f"::{self.parse_name_component()}"
        else:
            type_name = self.current_token[1]
            self.eat(token_type)

        if token_type == "LONG" and self.current_token[0] == "LONG":
            type_name += f" {self.current_token[1]}"
            self.eat("LONG")
            if self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                type_name += f" {self.current_token[1]}"
                self.eat(self.current_token[0])
            if self.current_token[0] == "INT":
                type_name += f" {self.current_token[1]}"
                self.eat("INT")
        elif (
            token_type in self.COMPOSITE_SCALAR_TYPE_SUFFIX_TOKENS
            and self.current_token[0]
            in self.COMPOSITE_SCALAR_TYPE_SUFFIX_TOKENS[token_type]
        ):
            type_name += f" {self.current_token[1]}"
            self.eat(self.current_token[0])

        while True:
            if self.current_token[0] == "LESS_THAN":
                type_name += self.parse_template_suffix()
                continue
            if self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                if self.current_token[0] == "TEMPLATE":
                    self.eat("TEMPLATE")
                member = self.parse_name_component()
                type_name += f"::{member}"
                continue
            break

        return type_name

    def parse_elaborated_type_name(self):
        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        member = self.parse_name_component()
        return f"{type_name} {member}"

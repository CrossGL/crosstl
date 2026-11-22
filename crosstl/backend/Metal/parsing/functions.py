"""Function declarations in the Metal parser: defaulted and deleted functions, method qualifiers, operator suffixes, trailing return types, stage attributes, parameters and attributes."""

from crosstl.translator.refusals import MetalSyntaxError

from ..MetalAst import (
    AttributeNode,
    VariableNode,
)
from .parser_tables import STAGE_ATTRIBUTE_NAMES


class MetalFunctionParsingMixin:
    """Function members of ``MetalParser``."""

    def is_defaulted_or_deleted_function_declaration(self):
        return (
            self.current_token[0] == "EQUALS"
            and self.peek(1)[0] in {"DEFAULT", "IDENTIFIER"}
            and self.peek(1)[1] in {"default", "delete"}
            and self.peek(2)[0] == "SEMICOLON"
        )

    def skip_defaulted_or_deleted_function_declaration(self):
        self.eat("EQUALS")
        self.eat(self.current_token[0])
        self.eat("SEMICOLON")

    def parse_function_method_qualifiers(self):
        qualifiers = []
        while self.current_token[0] in {
            "CONST",
            "VOLATILE",
            "BITWISE_AND",
            "AND",
            "CONSTANT",
            "DEVICE",
            "THREAD",
            "THREADGROUP",
            "THREADGROUP_IMAGEBLOCK",
        }:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        if self.current_token == ("IDENTIFIER", "noexcept"):
            qualifiers.append(self.current_token[1])
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                qualifiers.append(
                    self.parse_balanced_token_text(
                        "LPAREN",
                        "RPAREN",
                        error_message="Unterminated noexcept qualifier",
                    )
                )

        while self.current_token == (
            "IDENTIFIER",
            "override",
        ) or self.current_token == (
            "IDENTIFIER",
            "final",
        ):
            qualifiers.append(self.current_token[1])
            self.eat("IDENTIFIER")
        return qualifiers

    def parse_function_operator_suffix(self, name):
        if (
            name.split("::")[-1] != "operator"
            or self.current_token[0] != "LPAREN"
            or self.peek(1)[0] != "RPAREN"
        ):
            return name
        self.eat("LPAREN")
        self.eat("RPAREN")
        return f"{name}()"

    def parse_optional_trailing_return_type(self, attributes):
        if self.current_token[0] != "ARROW":
            return None
        self.eat("ARROW")
        trailing_attributes = self.parse_attributes()
        return_type, _return_qualifiers = self.parse_type_specifier(
            attributes=trailing_attributes
        )
        attributes.extend(trailing_attributes)
        return return_type

    def extract_stage_attributes(self, attributes):
        qualifier = None
        remaining = []
        for attr in attributes:
            attr_name = getattr(attr, "name", None)
            if attr_name in STAGE_ATTRIBUTE_NAMES and not getattr(attr, "args", []):
                if qualifier is None:
                    qualifier = attr_name
                continue
            remaining.append(attr)
        return qualifier, remaining

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            if self.current_token[0] == "ELLIPSIS":
                self.eat("ELLIPSIS")
                params.append(VariableNode("...", ""))
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                if self.current_token[0] == "RPAREN":
                    break
                raise MetalSyntaxError(
                    f"Expected comma or closing parenthesis, got {self.current_token[0]}"
                )
            attributes = self.parse_attributes()
            vtype, qualifiers = self.parse_type_specifier(attributes=attributes)
            qualifier_contract = dict(self.last_type_specifier_qualifier_contract)
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            if grouped_suffix and self.current_token[0] == "LPAREN":
                self.parse_function_pointer_parameter_suffix()
            param_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            attributes.extend(self.parse_attributes())
            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()
            param_attributes = self.parse_attributes()
            attributes.extend(param_attributes)

            var_node = VariableNode(
                param_type, name, qualifiers=qualifiers, attributes=attributes
            )
            var_node.pointee_qualifiers = list(qualifier_contract["pointee"])
            var_node.indirection_qualifiers = list(qualifier_contract["indirection"])
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.default_value = default_value
            params.append(var_node)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] == "RPAREN":
                break
            else:
                raise MetalSyntaxError(
                    f"Expected comma or closing parenthesis, got {self.current_token[0]}"
                )
        return params

    def parse_function_pointer_parameter_suffix(self):
        self.parse_balanced_token_text(
            "LPAREN",
            "RPAREN",
            error_message="Unterminated function pointer parameter",
        )

    def is_gnu_attribute_start(self):
        return (
            self.current_token == ("IDENTIFIER", "__attribute__")
            and self.peek(1)[0] == "LPAREN"
        )

    def is_gnu_attribute_start_at(self, idx):
        return (
            idx + 1 < len(self.tokens)
            and self.tokens[idx] == ("IDENTIFIER", "__attribute__")
            and self.tokens[idx + 1][0] == "LPAREN"
        )

    def skip_balanced_tokens_at(self, idx, open_token, close_token):
        if idx >= len(self.tokens) or self.tokens[idx][0] != open_token:
            return idx
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                if depth == 0:
                    return idx + 1
            elif (
                open_token == "LESS_THAN"
                and close_token == "GREATER_THAN"
                and token_type == "SHIFT_RIGHT"
                and depth >= 2
            ):
                depth -= 2
                if depth == 0:
                    return idx + 1
            idx += 1
        return idx

    def skip_leading_attribute_tokens_at(self, idx):
        while idx < len(self.tokens):
            if self.tokens[idx][0] == "ATTRIBUTE":
                idx += 1
            elif self.is_gnu_attribute_start_at(idx):
                idx = self.skip_balanced_tokens_at(idx + 1, "LPAREN", "RPAREN")
            else:
                break
        return idx

    def parse_attributes(self):
        def split_top_level(text):
            parts = []
            buf = ""
            depth = 0
            for ch in text:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                if ch == "," and depth == 0:
                    if buf.strip():
                        parts.append(buf.strip())
                    buf = ""
                    continue
                buf += ch
            if buf.strip():
                parts.append(buf.strip())
            return parts

        def append_attributes(attr_content):
            for part in split_top_level(attr_content):
                name = part
                args = []
                if "(" in part and part.endswith(")"):
                    name, arg_str = part.split("(", 1)
                    arg_str = arg_str[:-1]  # remove trailing )
                    args = [arg.strip() for arg in split_top_level(arg_str)]
                    name = name.strip()
                attributes.append(AttributeNode(name.strip(), args))

        attributes = []
        while self.current_token[0] == "ATTRIBUTE" or self.is_gnu_attribute_start():
            if self.current_token[0] == "ATTRIBUTE":
                attr_content = self.current_token[1][2:-2].strip()  # Remove [[ and ]]
                append_attributes(attr_content)
                self.eat("ATTRIBUTE")
                continue

            self.eat("IDENTIFIER")
            attr_content = self.parse_balanced_token_text(
                "LPAREN", "RPAREN", "Unterminated GNU attribute"
            ).strip()
            if attr_content.startswith("(") and attr_content.endswith(")"):
                attr_content = attr_content[1:-1].strip()
            append_attributes(attr_content)
        return attributes

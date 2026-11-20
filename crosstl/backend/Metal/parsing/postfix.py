"""Postfix expressions, conversion-operator calls, cast targets and template argument suffixes in the Metal parser."""

import re

from ..MetalAst import (
    ArrayAccessNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    PostfixOpNode,
    UnaryOpNode,
    VariableNode,
)
from .parser_tables import (
    CAST_TYPE_QUALIFIER_NAMES,
    OPERATOR_OVERLOAD_TOKENS,
    TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS,
    TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS,
    TYPE_TOKENS,
)


class MetalPostfixParsingMixin:
    """Postfix members of ``MetalParser``."""

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.is_static_cast_template_call(node):
                template_args = self.parse_template_argument_suffix()
                target_type, qualifiers = self.split_cast_target_qualifiers(
                    self.format_generic_type_tokens(template_args)
                )
                self.eat("LPAREN")
                expression = self.parse_expression()
                self.eat("RPAREN")
                node = CastNode(target_type, expression, qualifiers=qualifiers)
                continue
            if self.is_as_type_template_call(node):
                template_args = self.parse_template_argument_suffix()
                suffix = f"<{''.join(template_args)}>"
                if isinstance(node, VariableNode):
                    node.name += suffix
                else:
                    node.member += suffix
                continue
            if self.is_template_identifier_suffix(node):
                template_args = self.parse_template_argument_suffix()
                suffix = f"<{self.format_generic_type_tokens(template_args)}>"
                if isinstance(node, VariableNode):
                    node.name += suffix
                else:
                    node.member += suffix
                continue
            if self.is_template_variable_suffix(node):
                template_args = self.parse_template_argument_suffix()
                suffix = f"<{self.format_generic_type_tokens(template_args)}>"
                if isinstance(node, VariableNode):
                    node.name += suffix
                else:
                    node.member += suffix
                continue
            if self.current_token[0] == "LPAREN":
                node = self.parse_call(node)
                continue
            if self.current_token[0] == "ELLIPSIS":
                self.eat("ELLIPSIS")
                node = UnaryOpNode("post...", node)
                continue
            if self.current_token[0] == "LBRACE" and isinstance(node, VariableNode):
                initializer = self.parse_initializer_list()
                node = FunctionCallNode(node.name, [initializer])
                node.is_braced_constructor = True
                continue
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                self.skip_template_disambiguator()
                if self.is_conversion_operator_call_start():
                    node = self.parse_conversion_operator_call(node)
                    continue
                member = self.parse_member_name("dot")
                node = MemberAccessNode(node, member)
                continue
            if self.current_token[0] == "ARROW":
                self.eat("ARROW")
                self.skip_template_disambiguator()
                if self.is_conversion_operator_call_start():
                    node = self.parse_conversion_operator_call(node)
                    continue
                member = self.parse_member_name("arrow")
                node = MemberAccessNode(node, member, True)
                continue
            if self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                member = self.parse_member_name("scope")
                if isinstance(node, VariableNode):
                    node.name += f"::{member}"
                elif isinstance(node, MemberAccessNode):
                    node.member += f"::{member}"
                else:
                    node = MemberAccessNode(node, member)
                continue
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = None
                if self.current_token[0] != "RBRACKET":
                    index = self.parse_expression()
                self.eat("RBRACKET")
                node = ArrayAccessNode(node, index)
                continue
            if self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                node = PostfixOpNode(node, op)
                continue
            break
        return node

    def is_conversion_operator_call_start(self):
        if self.current_token != ("IDENTIFIER", "operator"):
            return False
        next_type = self.peek(1)[0]
        return next_type not in OPERATOR_OVERLOAD_TOKENS

    def parse_conversion_operator_call(self, node):
        self.eat("IDENTIFIER")
        target_type, qualifiers = self.parse_type_specifier()
        self.eat("LPAREN")
        self.eat("RPAREN")
        return CastNode(target_type, node, qualifiers=qualifiers)

    @staticmethod
    def split_cast_target_qualifiers(target_type):
        remaining = str(target_type).strip()
        qualifiers = []
        while remaining:
            match = re.match(r"([A-Za-z_]\w*)\b\s*", remaining)
            if match is None or match.group(1) not in CAST_TYPE_QUALIFIER_NAMES:
                break
            qualifiers.append(match.group(1))
            remaining = remaining[match.end() :]
        return remaining, qualifiers

    def skip_template_disambiguator(self):
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "template"
            and self.peek(1)[0] in {"IDENTIFIER", "READ", "WRITE", "READ_WRITE"}
        ):
            self.eat("IDENTIFIER")

    def is_static_cast_template_call(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if isinstance(node, VariableNode):
            callee_name = node.name
        elif isinstance(node, MemberAccessNode):
            callee_name = node.member
        else:
            return False
        if callee_name.split("::")[-1] not in {
            "static_cast",
            "reinterpret_cast",
            "const_cast",
        }:
            return False
        return self.template_argument_list_followed_by_call(
            require_type_like_argument=False
        )

    def is_as_type_template_call(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if isinstance(node, VariableNode):
            callee_name = node.name
        elif isinstance(node, MemberAccessNode):
            callee_name = node.member
        else:
            return False
        if callee_name.split("::")[-1] != "as_type":
            return False
        return self.template_argument_list_followed_by_call()

    def is_template_identifier_suffix(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if not isinstance(node, (VariableNode, MemberAccessNode)):
            return False
        name = node.name if isinstance(node, VariableNode) else node.member
        if name in self.known_function_templates:
            return self.template_argument_list_followed_by_call(
                follow_token_types=TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS,
                require_type_like_argument=False,
            )
        if self.template_argument_list_has_numeric_value_expression(self.pos):
            return self.template_argument_list_followed_by_call(
                follow_token_types=TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS,
                require_type_like_argument=False,
            )
        return self.template_argument_list_followed_by_call(
            follow_token_types=TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS,
            require_type_like_argument=True,
        )

    def is_template_variable_suffix(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if not isinstance(node, (VariableNode, MemberAccessNode)):
            return False
        name = node.name if isinstance(node, VariableNode) else node.member
        if not (
            self.is_variable_template_name(name)
            or name in self.known_variable_templates
        ):
            return False
        return self.template_argument_list_followed_by_call(
            follow_token_types=TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS,
            require_type_like_argument=True,
        )

    def template_argument_list_has_numeric_value_expression(self, idx):
        if idx >= len(self.tokens) or self.tokens[idx][0] != "LESS_THAN":
            return False
        depth = 0
        grouped_depth = 0
        saw_value_token = False
        while idx < len(self.tokens):
            token_type, token_value = self.tokens[idx]
            if depth > 0 and token_type in {"SEMICOLON", "LBRACE", "RBRACE", "EOF"}:
                return False
            grouping = self.template_argument_grouping(token_type, depth, grouped_depth)
            if grouping is None:
                return False
            nested = grouping or grouped_depth
            grouped_depth = grouping
            if not nested and token_type == "LESS_THAN":
                depth += 1
            elif not nested and token_type == "SHIFT_RIGHT" and depth >= 2:
                depth -= 2
                if depth == 0:
                    return saw_value_token
            elif not nested and token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    return saw_value_token
            elif depth > 0:
                if token_type in {"NUMBER", "TRUE", "FALSE"} or (
                    token_type == "IDENTIFIER"
                    and (
                        token_value in self.known_value_template_parameters
                        or self.is_known_local_variable_name(token_value)
                    )
                ):
                    saw_value_token = True
            idx += 1
        return False

    @staticmethod
    def template_argument_grouping(token_type, depth, grouped_depth):
        """Return the parenthesis depth after ``token_type`` inside a template argument list.

        A ``<`` or ``>`` nested in parentheses or brackets neither opens nor closes the list (C++
        [temp.names]), and a closer with no opener inside the list means the ``<`` was a comparison, as in
        ``i < ((a) > (b) ? (a) : (b))``; that case returns ``None``.
        """
        if depth == 0:
            return grouped_depth
        if token_type in {"LPAREN", "LBRACKET"}:
            return grouped_depth + 1
        if token_type in {"RPAREN", "RBRACKET"}:
            return grouped_depth - 1 if grouped_depth else None
        return grouped_depth

    def is_variable_template_name(self, name):
        if not isinstance(name, str):
            return False
        return name.split("::")[-1].endswith("_v")

    def template_argument_list_followed_by_call(
        self, follow_token_types=None, require_type_like_argument=False
    ):
        if self.current_token[0] != "LESS_THAN":
            return False
        follow_token_types = follow_token_types or {"LPAREN"}
        idx = self.pos
        depth = 0
        grouped_depth = 0
        saw_type_like = False
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if depth > 0 and token_type in {"SEMICOLON", "LBRACE", "RBRACE", "EOF"}:
                return False
            grouping = self.template_argument_grouping(token_type, depth, grouped_depth)
            if grouping is None:
                return False
            nested = grouping or grouped_depth
            grouped_depth = grouping
            if not nested and token_type == "LESS_THAN":
                depth += 1
            elif not nested and token_type == "SHIFT_RIGHT" and depth >= 2:
                depth -= 2
                if depth == 0:
                    if require_type_like_argument and not saw_type_like:
                        return False
                    return (
                        idx + 1 < len(self.tokens)
                        and self.tokens[idx + 1][0] in follow_token_types
                    )
            elif not nested and token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    if require_type_like_argument and not saw_type_like:
                        return False
                    return (
                        idx + 1 < len(self.tokens)
                        and self.tokens[idx + 1][0] in follow_token_types
                    )
            elif depth > 0 and self.is_template_type_argument_token(idx):
                saw_type_like = True
            idx += 1
        return False

    def is_template_type_argument_token(self, idx):
        token_type, token_value = self.tokens[idx]
        if token_type in TYPE_TOKENS - {"IDENTIFIER", "TYPEDEF"}:
            return True
        if token_type == "IDENTIFIER":
            next_type = self.tokens[idx + 1][0] if idx + 1 < len(self.tokens) else "EOF"
            return token_value in self.known_types or next_type in {
                "SCOPE",
                "LESS_THAN",
                "COMMA",
                "GREATER_THAN",
            }
        return False

    def parse_template_argument_suffix(self):
        self.eat("LESS_THAN")
        depth = 1
        grouped_depth = 0
        parts = []
        while depth > 0 and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if grouped_depth == 0 and token_type == "LESS_THAN":
                depth += 1
                parts.append(token_value)
                self.eat("LESS_THAN")
            elif grouped_depth == 0 and token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parts.append(token_value)
                self.eat("GREATER_THAN")
            elif grouped_depth == 0 and token_type == "SHIFT_RIGHT" and depth >= 2:
                parts.extend(">" for _ in range(min(2, depth - 1)))
                depth -= 2
                self.eat("SHIFT_RIGHT")
                if depth == 0:
                    break
            else:
                if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                    grouped_depth += 1
                elif (
                    token_type in {"RPAREN", "RBRACKET", "RBRACE"} and grouped_depth > 0
                ):
                    grouped_depth -= 1
                parts.append(token_value)
                self.eat(token_type)
        return parts

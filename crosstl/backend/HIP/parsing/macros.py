"""Test-framework and statement macros, newline skipping and type-token predicates in the HIP parser."""

import re

from ..HipAst import (
    FunctionNode,
    HipBuiltinNode,
    VariableNode,
)


class HipMacroMixin:
    """Macro members of ``HipParser``."""

    def is_hip_dynamic_shared_macro(self):
        return (
            self.match("IDENTIFIER")
            and self.current_token.value == "HIP_DYNAMIC_SHARED"
            and self.peek_non_newline()
            and self.peek_non_newline().type == "LPAREN"
        )

    def parse_hip_dynamic_shared_macro(self):
        self.advance()
        self.consume("LPAREN")
        self.skip_newlines()
        var_type = self.parse_type()
        self.skip_newlines()
        self.consume("COMMA")
        self.skip_newlines()
        name = self.consume_declarator_name()
        self.skip_newlines()
        self.consume("RPAREN")
        if self.match("SEMICOLON"):
            self.advance()

        qualifiers = ["extern", "__shared__"]
        return VariableNode(
            f"{var_type}[]",
            name,
            qualifiers=qualifiers,
            is_extern_shared_memory=True,
            is_dynamic_shared_memory=True,
        )

    def is_catch_test_block_macro(self):
        if self.block_depth != 0:
            return False
        if not (
            self.match("IDENTIFIER")
            and self.current_token.value in self.CATCH_TEST_BLOCK_MACRO_NAMES
        ):
            return False

        index = self.skip_newlines_at_pos(self.pos + 1)
        if index >= len(self.tokens) or self.tokens[index].type != "LPAREN":
            return False

        index = self.skip_balanced_group_at_pos(index, "LPAREN", "RPAREN")
        if index is None:
            return False
        index = self.skip_newlines_at_pos(index)
        return index < len(self.tokens) and self.tokens[index].type == "LBRACE"

    def parse_catch_test_block_macro(self):
        macro_name = self.consume("IDENTIFIER").value
        args = self.consume_raw_macro_arguments()
        self.skip_newlines()
        body = self.parse_block()
        test_name = self.catch_test_function_name(macro_name, args)
        return FunctionNode("void", test_name, [], body, qualifiers=[macro_name])

    def is_catch_nested_block_macro(self):
        if self.block_depth == 0:
            return False
        if not (
            self.match("IDENTIFIER")
            and self.current_token.value in self.CATCH_NESTED_BLOCK_MACRO_NAMES
        ):
            return False

        index = self.skip_newlines_at_pos(self.pos + 1)
        if index >= len(self.tokens) or self.tokens[index].type != "LPAREN":
            return False

        index = self.skip_balanced_group_at_pos(index, "LPAREN", "RPAREN")
        if index is None:
            return False
        index = self.skip_newlines_at_pos(index)
        return index < len(self.tokens) and self.tokens[index].type == "LBRACE"

    def parse_catch_nested_block_macro(self):
        self.consume("IDENTIFIER")
        self.consume_raw_macro_arguments()
        self.skip_newlines()
        return self.parse_block()

    def is_bare_nested_block_macro(self):
        if self.block_depth == 0:
            return False
        if not (
            self.match("IDENTIFIER")
            and self.current_token.value in self.BARE_NESTED_BLOCK_MACRO_NAMES
        ):
            return False

        index = self.skip_newlines_at_pos(self.pos + 1)
        return index < len(self.tokens) and self.tokens[index].type == "LBRACE"

    def parse_bare_nested_block_macro(self):
        self.consume("IDENTIFIER")
        self.skip_newlines()
        return self.parse_block()

    def is_standalone_statement_macro(self):
        if not (
            self.match("IDENTIFIER")
            and self.current_token.value in self.STANDALONE_STATEMENT_MACRO_NAMES
        ):
            return False

        next_index = self.pos + 1
        return next_index >= len(self.tokens) or self.tokens[next_index].type in {
            "NEWLINE",
            "SEMICOLON",
            "RBRACE",
        }

    def parse_standalone_statement_macro(self):
        self.advance()
        self.skip_newlines()
        if self.match("SEMICOLON"):
            self.advance()
        return None

    def consume_raw_macro_arguments(self):
        self.consume("LPAREN")
        args = [[]]
        depth = 1

        while self.current_token and depth > 0:
            token = self.current_token

            if token.type == "LPAREN":
                depth += 1
                args[-1].append(token)
                self.advance()
                continue

            if token.type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.advance()
                    break
                args[-1].append(token)
                self.advance()
                continue

            if token.type == "COMMA" and depth == 1:
                args.append([])
                self.advance()
                continue

            args[-1].append(token)
            self.advance()

        if depth != 0:
            self.error("Unterminated macro argument list")

        return args

    def catch_test_function_name(self, macro_name, args):
        if macro_name in self.GTEST_TEST_BLOCK_MACRO_NAMES and len(args) >= 2:
            raw_name = "_".join(self.macro_argument_text(arg) for arg in args[:2])
        elif args and args[0]:
            raw_name = self.macro_argument_text(args[0])
        else:
            raw_name = macro_name.lower()

        raw_name = raw_name.strip()
        if (
            len(raw_name) >= 2
            and raw_name[0] == raw_name[-1]
            and raw_name[0]
            in {
                '"',
                "'",
            }
        ):
            raw_name = raw_name[1:-1]

        name = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_")
        if not name:
            name = macro_name.lower()
        if name[0].isdigit():
            name = f"test_{name}"
        return name

    def macro_argument_text(self, tokens):
        return " ".join(token.value for token in tokens if token.type != "NEWLINE")

    def skip_balanced_group_at_pos(self, index, open_type, close_type):
        if index >= len(self.tokens) or self.tokens[index].type != open_type:
            return None

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == open_type:
                depth += 1
            elif token_type == close_type:
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1

        return None

    def parse_flat_builtin_node(self):
        token = self.current_token
        builtin_name = self.FLAT_BUILTIN_TOKEN_MAP[token.type]
        component = token.value.rsplit("_", 1)[-1]
        self.advance()
        return HipBuiltinNode(builtin_name, component)

    def skip_newlines(self):
        while self.match("NEWLINE"):
            self.advance()

    def skip_newlines_at_pos(self, index):
        while index < len(self.tokens) and self.tokens[index].type == "NEWLINE":
            index += 1
        return index

    def peek_non_newline(self, offset=1):
        index = self.pos + offset
        index = self.skip_newlines_at_pos(index)
        if index < len(self.tokens):
            return self.tokens[index]
        return None

    def is_type_constructor_expression_start(self):
        next_token = self.peek_non_newline()
        return (
            self.is_type_token(allow_identifier=False)
            and next_token
            and next_token.type in {"LPAREN", "LBRACE"}
        )

    def is_sign_only_functional_cast(self):
        next_token = self.peek_non_newline()
        return next_token is not None and next_token.type == "LPAREN"

    def is_builtin_type_token(self, token=None):
        token = token or self.current_token
        if not token:
            return False
        if token.type == "FLOAT":
            return token.value == "float"
        if token.type == "CHAR":
            return token.value == "char"
        return token.type in self.BUILTIN_TYPE_TOKENS

    def is_type_token(self, token=None, allow_identifier=True):
        token = token or self.current_token
        if not token:
            return False
        if self.is_builtin_type_token(token):
            return True
        if token.type in self.VECTOR_TYPE_TOKENS:
            return True
        if token.type in self.RESOURCE_TYPE_TOKENS:
            return True
        if token.type in self.ELABORATED_TYPE_TOKENS:
            return True
        return allow_identifier and token.type == "IDENTIFIER"

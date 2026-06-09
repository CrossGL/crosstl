"""OpenCL parser for converting OpenCL C tokens to AST."""

import re

from crosstl.backend.HIP.HipAst import (
    ForNode,
    FunctionCallNode,
    FunctionNode,
    KernelNode,
)
from crosstl.backend.HIP.HipParser import HipParser

from .OpenCLAst import OpenCLProgramNode, OpenCLStatementExpressionNode
from .OpenCLLexer import OpenCLLexer


class OpenCLParser(HipParser):
    """Parse OpenCL C tokens into the OpenCL backend AST."""

    FUNCTION_DECLARATION_SPECIFIER_TOKENS = {
        *HipParser.FUNCTION_DECLARATION_SPECIFIER_TOKENS,
        "__GLOBAL__",
    }
    OPENCL_POST_RETURN_FUNCTION_SPECIFIER_TOKENS = {"__GLOBAL__", "INLINE"}
    IDENTIFIER_FUNCTION_SPECIFIER_VALUES = {
        *HipParser.IDENTIFIER_FUNCTION_SPECIFIER_VALUES,
        "__inline",
        "__inline__",
        "INLINE_FUNC",
        "OPENCL_KERNEL",
    }
    KERNEL_FUNCTION_SPECIFIER_VALUES = {"kernel", "__kernel", "OPENCL_KERNEL"}
    DECLARATION_QUALIFIER_TOKENS = {
        *HipParser.DECLARATION_QUALIFIER_TOKENS,
        "__DEVICE__",
        "__SHARED__",
        "__CONSTANT__",
        "__MANAGED__",
    }
    TYPE_QUALIFIER_TOKENS = {
        *HipParser.TYPE_QUALIFIER_TOKENS,
        "__DEVICE__",
        "__SHARED__",
        "__CONSTANT__",
        "__MANAGED__",
        "READ_ONLY",
        "WRITE_ONLY",
        "READ_WRITE",
    }
    POSTFIX_TYPE_QUALIFIER_TOKENS = {
        *HipParser.POSTFIX_TYPE_QUALIFIER_TOKENS,
        "__DEVICE__",
        "__SHARED__",
        "__CONSTANT__",
        "__MANAGED__",
        "READ_ONLY",
        "WRITE_ONLY",
        "READ_WRITE",
    }
    TYPE_ATTRIBUTE_IDENTIFIERS = {
        *HipParser.TYPE_ATTRIBUTE_IDENTIFIERS,
        "__attribute__",
    }
    ATOMIC_FUNCTION_NAMES = {
        *HipParser.ATOMIC_FUNCTION_NAMES,
        "atomic_add",
        "atomic_sub",
        "atomic_xchg",
        "atomic_inc",
        "atomic_dec",
        "atomic_cmpxchg",
        "atomic_min",
        "atomic_max",
        "atomic_and",
        "atomic_or",
        "atomic_xor",
        "atomic_fetch_add",
        "atomic_fetch_sub",
        "atomic_fetch_min",
        "atomic_fetch_max",
        "atomic_fetch_and",
        "atomic_fetch_or",
        "atomic_fetch_xor",
        "atomic_fetch_add_explicit",
        "atomic_fetch_sub_explicit",
        "atomic_fetch_min_explicit",
        "atomic_fetch_max_explicit",
        "atomic_fetch_and_explicit",
        "atomic_fetch_or_explicit",
        "atomic_fetch_xor_explicit",
        "atomic_exchange",
        "atomic_exchange_explicit",
        "atomic_compare_exchange_strong",
        "atomic_compare_exchange_weak",
        "atomic_compare_exchange_strong_explicit",
        "atomic_compare_exchange_weak_explicit",
    }
    PACK_EXPANSION_FUNCTION_NAME = "__opencl_pack_expand__"
    OPENCL_IDENTIFIER_TYPE_NAMES = {
        "char",
        "uchar",
        "ushort",
        "uint",
        "ulong",
        "intptr_t",
        "uintptr_t",
        "ptrdiff_t",
        "size_t",
        "half",
        "dim_t",
        "sampler_t",
        "event_t",
        "clk_event_t",
        "image1d_t",
        "image1d_array_t",
        "image1d_buffer_t",
        "image2d_t",
        "image2d_array_t",
        "image2d_depth_t",
        "image2d_array_depth_t",
        "image3d_t",
    }
    OPENCL_VECTOR_CONSTRUCTOR_QUALIFIERS = {
        "const",
        "volatile",
        "__restrict__",
        "restrict",
        "__global__",
        "__shared__",
        "__local",
        "__local__",
        "__constant",
        "__constant__",
        "__private",
        "__private__",
        "global",
        "local",
        "constant",
        "private",
    }
    NON_CAST_FOLLOW_TOKENS = {
        "ASSIGN",
        "PLUS_ASSIGN",
        "MINUS_ASSIGN",
        "MULTIPLY_ASSIGN",
        "DIVIDE_ASSIGN",
        "STAR_ASSIGN",
        "SLASH_ASSIGN",
        "PERCENT_ASSIGN",
        "AND_ASSIGN",
        "OR_ASSIGN",
        "XOR_ASSIGN",
        "LSHIFT_ASSIGN",
        "RSHIFT_ASSIGN",
        "LT",
        "LE",
        "GT",
        "GE",
        "EQ",
        "NE",
        "AND",
        "OR",
        "LOGICAL_AND",
        "LOGICAL_OR",
        "PIPE",
        "XOR",
        "QUESTION",
        "COLON",
        "COMMA",
        "RPAREN",
        "RBRACKET",
        "RBRACE",
        "SEMICOLON",
    }

    def parse(self):
        statements = []

        while self.current_token:
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    statements.extend(stmt)
                else:
                    statements.append(stmt)

        return OpenCLProgramNode(statements)

    def parse_expression_statement(self):
        expressions = [self.parse_expression()]
        self.skip_newlines()

        while self.match("COMMA"):
            self.advance()
            self.skip_newlines()
            expressions.append(self.parse_expression())
            self.skip_newlines()

        if self.match("SEMICOLON"):
            self.advance()
            return expressions if len(expressions) > 1 else expressions[0]

        expr = expressions[0]
        if len(expressions) == 1 and self.is_optional_semicolon_macro_statement(expr):
            self.skip_newlines()
            if self.match("SEMICOLON"):
                self.advance()
            return expr

        self.skip_newlines()
        self.consume("SEMICOLON")

        return expressions if len(expressions) > 1 else expressions[0]

    def is_function_qualifier_token_at_pos(self, index):
        if index >= len(self.tokens):
            return False

        token = self.tokens[index]
        return (
            token.type in self.FUNCTION_DECLARATION_SPECIFIER_TOKENS
            or token.type in self.OPENCL_POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or token.type == "__LAUNCH_BOUNDS__"
            or token.type == "CONSTEXPR"
            or self.is_identifier_function_specifier_token(token)
        )

    def function_name_index_at(self, index):
        linkage_end = self.skip_linkage_specifier_at_pos(index)
        if linkage_end is not None:
            index = linkage_end

        index = self.skip_newlines_at_pos(index)
        index = self.skip_cpp_attributes_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)

        index = self.skip_function_specifiers_at_pos(index)
        index = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        if index is None:
            return None

        index = self.skip_newlines_at_pos(index)
        index = self.skip_post_return_function_specifiers_at_pos(index)

        if self.skip_function_name_at(index) is not None:
            return index

        return None

    def is_function_declaration(self) -> bool:
        index = self.pos

        index = self.skip_cpp_attributes_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)
        index = self.skip_function_specifiers_at_pos(index)
        index = self.skip_type_at_pos(
            index, allow_unknown_identifier_pointers=self.block_depth == 0
        )
        if index is not None:
            index = self.skip_newlines_at_pos(index)
            index = self.skip_post_return_function_specifiers_at_pos(index)
            parameter_list_start = self.skip_function_name_at(index)
            if (
                parameter_list_start is not None
                and self.is_plausible_function_parameter_list_at_pos(
                    parameter_list_start
                )
            ):
                return True

        return False

    def skip_function_specifiers_at_pos(self, index):
        while index < len(self.tokens) and (
            self.tokens[index].type in self.FUNCTION_DECLARATION_SPECIFIER_TOKENS
            or self.is_identifier_function_specifier_token(self.tokens[index])
        ):
            index += 1
            index = self.skip_newlines_at_pos(index)
            index = self.skip_cpp_attributes_at_pos(index)
            index = self.skip_type_attribute_prefixes_at_pos(index)
        return index

    def skip_post_return_function_specifiers_at_pos(self, index):
        while index < len(self.tokens) and (
            self.tokens[index].type in self.OPENCL_POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_identifier_function_specifier_token(self.tokens[index])
        ):
            index += 1
            index = self.skip_newlines_at_pos(index)
        return index

    def consume_function_name(self):
        name = super().consume_function_name()
        self.skip_newlines()
        return name

    def skip_function_name_at(self, index):
        if index >= len(self.tokens) or not self.is_function_name_token(
            self.tokens[index]
        ):
            return None

        name = self.tokens[index].value
        index += 1
        index = self.skip_operator_function_suffix_at(index, name)
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        while index < len(self.tokens) and self.tokens[index].type == "SCOPE":
            index += 1
            if index < len(self.tokens) and self.tokens[index].type == "TILDE":
                index += 1
            if index >= len(self.tokens) or not self.is_function_name_token(
                self.tokens[index]
            ):
                return None
            name = self.tokens[index].value
            index += 1
            index = self.skip_operator_function_suffix_at(index, name)
            if index < len(self.tokens) and self.tokens[index].type == "LT":
                index = self.skip_template_at_pos(index)
                if index is None:
                    return None

        index = self.skip_newlines_at_pos(index)
        if index < len(self.tokens) and self.tokens[index].type == "LPAREN":
            return index
        return None

    def parse_variable_declaration_type(self, qualifiers):
        saved_pos = self.pos
        saved_token = self.current_token
        type_prefixes = []
        saw_integral_sign = False

        self.skip_newlines()
        self.skip_cpp_attributes()
        self.parse_type_attribute_prefixes()

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_prefixes.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        self.skip_cpp_attributes()
        interleaved_qualifiers = self.parse_declaration_qualifiers()
        if type_prefixes or interleaved_qualifiers:
            qualifiers.extend(interleaved_qualifiers)
            self.skip_newlines()
            if self.is_implicit_int_current_type(saw_integral_sign):
                base_type = "int"
            else:
                base_type = self.parse_type()
            qualifiers.extend(self.parse_declaration_qualifiers())
            return " ".join([*type_prefixes, base_type]).strip()

        self.pos = saved_pos
        self.current_token = saved_token
        self.skip_newlines()
        self.skip_cpp_attributes()
        self.parse_type_attribute_prefixes()
        base_type = self.parse_type()
        qualifiers.extend(self.parse_declaration_qualifiers())
        return base_type

    def parse_variable_declarator(self, base_type, qualifiers, allow_prefix):
        if allow_prefix:
            base_type = self.parse_declarator_prefix(base_type)
        self.skip_newlines()
        self.parse_type_attribute_prefixes()
        return super().parse_variable_declarator(
            base_type, qualifiers, allow_prefix=False
        )

    def parse_function_pointer_parameter_declarator(self):
        if not (
            self.match("LPAREN")
            and self.peek()
            and self.peek().type in {"ASTERISK", "STAR"}
        ):
            return None

        self.consume("LPAREN")
        self.advance()

        while self.match(*self.POSTFIX_TYPE_QUALIFIER_TOKENS):
            self.advance()

        name = ""
        if self.is_declarator_name_token():
            name = self.current_token.value
            self.advance()

        self.consume("RPAREN")

        if self.match("LPAREN"):
            self.skip_balanced_parentheses()

        return name

    def parse_simple_function(self):
        qualifiers = []
        attributes = []
        self.skip_cpp_attributes()
        attributes.extend(self.parse_type_attribute_prefixes())
        while (
            self.match(*self.FUNCTION_DECLARATION_SPECIFIER_TOKENS)
            or self.is_identifier_function_specifier_token()
        ):
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_cpp_attributes()
            attributes.extend(self.parse_type_attribute_prefixes())

        return_type = self.parse_type()
        self.skip_newlines()
        while (
            self.match(*self.OPENCL_POST_RETURN_FUNCTION_SPECIFIER_TOKENS)
            or self.is_identifier_function_specifier_token()
        ):
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        name = self.consume_function_name()
        self.user_function_names.add(name)
        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers(attributes)
        return_type = self.parse_trailing_return_type(return_type)

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        if any(item in self.KERNEL_FUNCTION_SPECIFIER_VALUES for item in qualifiers):
            return KernelNode(return_type, name, params, body, attributes)

        return FunctionNode(return_type, name, params, body, qualifiers, attributes)

    def parse_primary_expression(self):
        if self.is_opencl_statement_expression_start():
            return self.parse_opencl_statement_expression()
        if self.is_opencl_vector_constructor_cast():
            return self.parse_opencl_vector_constructor_cast()
        return super().parse_primary_expression()

    def is_opencl_statement_expression_start(self):
        if not self.match("LPAREN"):
            return False

        block_start = self.skip_newlines_at_pos(self.pos + 1)
        return (
            block_start < len(self.tokens) and self.tokens[block_start].type == "LBRACE"
        )

    def parse_opencl_statement_expression(self):
        self.consume("LPAREN")
        self.skip_newlines()
        statements = self.parse_block()
        self.skip_newlines()
        self.consume("RPAREN")
        return OpenCLStatementExpressionNode(statements)

    def parse_unary_expression(self):
        self.skip_newlines()
        return super().parse_unary_expression()

    def parse_for_statement(self):
        self.consume("FOR")
        self.skip_newlines()
        self.consume("LPAREN")
        self.skip_newlines()

        if self.is_range_for_statement():
            return self.parse_range_for_statement()

        init = None
        if not self.match("SEMICOLON"):
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list(
                    consume_semicolon=False
                )
                init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
                )
            else:
                init = self.parse_for_update_expression()
        self.skip_newlines()
        self.consume("SEMICOLON")
        self.skip_newlines()

        condition = None
        if not self.match("SEMICOLON"):
            condition = self.parse_expression()
        self.skip_newlines()
        self.consume("SEMICOLON")
        self.skip_newlines()

        update = None
        if not self.match("RPAREN"):
            update = self.parse_for_update_expression()
        self.skip_newlines()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return ForNode(init, condition, update, body)

    def is_opencl_vector_constructor_cast(self):
        if not self.match("LPAREN"):
            return False

        type_start = self.skip_newlines_at_pos(self.pos + 1)
        type_end = self.skip_type_at_pos(
            type_start,
            allow_unknown_identifier_pointers=True,
        )
        if (
            type_end is None
            or type_end + 1 >= len(self.tokens)
            or self.tokens[type_end].type != "RPAREN"
            or self.tokens[type_end + 1].type != "LPAREN"
        ):
            return False

        parts = [
            token.value
            for token in self.tokens[type_start:type_end]
            if token.type != "NEWLINE"
            and token.type
            not in {*self.TYPE_QUALIFIER_TOKENS, *self.POSTFIX_TYPE_QUALIFIER_TOKENS}
        ]
        if len(parts) != 1:
            return False
        if self.is_opencl_vector_type_name(parts[0]):
            return True
        return self.is_probable_identifier_type_name(
            parts[0]
        ) and self.has_top_level_comma_in_group_at_pos(type_end + 1)

    def parse_opencl_vector_constructor_cast(self):
        self.consume("LPAREN")
        target_type = self.parse_type()
        target_type = self.unqualified_opencl_vector_constructor_type(target_type)
        self.consume("RPAREN")
        self.consume("LPAREN")
        args = self.parse_argument_list()
        self.consume("RPAREN")
        return FunctionCallNode(target_type, args)

    def unqualified_opencl_vector_constructor_type(self, type_name):
        parts = [
            part
            for part in str(type_name).split()
            if part not in self.OPENCL_VECTOR_CONSTRUCTOR_QUALIFIERS
        ]
        if len(parts) == 1 and self.is_opencl_vector_type_name(parts[0]):
            return parts[0]
        return type_name

    def has_top_level_comma_in_group_at_pos(self, index):
        if index >= len(self.tokens) or self.tokens[index].type != "LPAREN":
            return False

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return False
            elif token_type == "COMMA" and depth == 1:
                return True
            index += 1

        return False

    def is_opencl_vector_type_name(self, type_name):
        return bool(
            re.fullmatch(
                r"(?:char|uchar|short|ushort|int|uint|long|ulong|float|double|half)"
                r"(?:2|3|4|8|16)",
                str(type_name),
            )
        )

    def is_cast_type_sequence(self, start, end):
        if super().is_cast_type_sequence(start, end):
            return True

        tokens = [token for token in self.tokens[start:end] if token.type != "NEWLINE"]
        index = 0
        saw_integral_sign = False
        while index < len(tokens) and tokens[index].type in self.TYPE_QUALIFIER_TOKENS:
            if tokens[index].type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        if not saw_integral_sign:
            return False

        return index < len(tokens) and all(
            token.type in {"ASTERISK", "STAR", *self.POSTFIX_TYPE_QUALIFIER_TOKENS}
            for token in tokens[index:]
        )

    def is_cast_expression(self):
        if not self.match("LPAREN"):
            return False

        type_start = self.skip_newlines_at_pos(self.pos + 1)
        type_end = self.skip_type_at_pos(
            type_start,
            allow_unknown_identifier_pointers=True,
        )
        if type_end is not None and type_end < len(self.tokens):
            after_type = self.skip_newlines_at_pos(type_end + 1)
            if (
                self.tokens[type_end].type == "RPAREN"
                and after_type < len(self.tokens)
                and self.tokens[after_type].type in self.NON_CAST_FOLLOW_TOKENS
            ):
                return False

        return super().is_cast_expression()

    def is_identifier_type_name(self, type_name):
        return (
            type_name in self.OPENCL_IDENTIFIER_TYPE_NAMES
            or super().is_identifier_type_name(type_name)
        )

    def is_probable_identifier_type_name(self, type_name):
        if super().is_probable_identifier_type_name(type_name):
            return True
        if not isinstance(type_name, str) or not type_name:
            return False
        return (
            type_name in {"T", "U"}
            or type_name.endswith("Type")
            or type_name.startswith(("real", "Real"))
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*T\d*", type_name)
            or type_name in {"Ti", "To", "Tk", "Tv", "accType"}
        )


def parse_opencl_code(code: str) -> OpenCLProgramNode:
    """Parse OpenCL source text and return the backend AST."""
    lexer = OpenCLLexer(code)
    tokens = lexer.tokenize()
    parser = OpenCLParser(tokens)
    return parser.parse()

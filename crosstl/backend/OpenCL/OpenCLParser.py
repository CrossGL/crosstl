"""OpenCL parser for converting OpenCL C tokens to AST."""

import re

from crosstl.backend.HIP.HipAst import FunctionCallNode, FunctionNode, KernelNode
from crosstl.backend.HIP.HipParser import HipParser

from .OpenCLAst import OpenCLProgramNode
from .OpenCLLexer import OpenCLLexer, Token


class OpenCLParser(HipParser):
    """Parse OpenCL C tokens into the OpenCL backend AST."""

    FUNCTION_DECLARATION_SPECIFIER_TOKENS = {
        *HipParser.FUNCTION_DECLARATION_SPECIFIER_TOKENS,
        "__GLOBAL__",
    }
    OPENCL_POST_RETURN_FUNCTION_SPECIFIER_TOKENS = {"__GLOBAL__"}
    IDENTIFIER_FUNCTION_SPECIFIER_VALUES = {
        *HipParser.IDENTIFIER_FUNCTION_SPECIFIER_VALUES,
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

    def parse_simple_function(self):
        qualifiers = []
        attributes = []
        self.skip_cpp_attributes()
        while (
            self.match(*self.FUNCTION_DECLARATION_SPECIFIER_TOKENS)
            or self.is_identifier_function_specifier_token()
        ):
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_cpp_attributes()

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
        if self.is_opencl_vector_constructor_cast():
            return self.parse_opencl_vector_constructor_cast()
        return super().parse_primary_expression()

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
        ]
        return len(parts) == 1 and self.is_opencl_vector_type_name(parts[0])

    def parse_opencl_vector_constructor_cast(self):
        self.consume("LPAREN")
        target_type = self.parse_type()
        self.consume("RPAREN")
        self.consume("LPAREN")
        args = self.parse_argument_list()
        self.consume("RPAREN")
        return FunctionCallNode(target_type, args)

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

        tokens = [
            token for token in self.tokens[start:end] if token.type != "NEWLINE"
        ]
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
            or type_name in {"Ti", "To", "Tk", "Tv", "accType"}
        )


def parse_opencl_code(code: str) -> OpenCLProgramNode:
    """Parse OpenCL source text and return the backend AST."""
    lexer = OpenCLLexer(code)
    tokens = lexer.tokenize()
    parser = OpenCLParser(tokens)
    return parser.parse()

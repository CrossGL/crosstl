"""Global and local variable declarations in the CUDA parser: prefixes, qualifiers, declarators, shared memory, initializers, bit-fields and array suffixes."""

import re

from ..CudaAst import (
    ConstantMemoryNode,
    FunctionCallNode,
    SharedMemoryNode,
    VariableNode,
)
from ..local_memory_spaces import refuse_local_memory_space


class CudaVariableDeclarationMixin:
    """Variable declaration members of ``CudaParser``."""

    def parse_global_variable(self):
        declarations = self.parse_variable_declaration_list()
        self.eat("SEMICOLON")
        return declarations if len(declarations) > 1 else declarations[0]

    def append_parsed_global_variable(self, target, variable):
        if isinstance(variable, list):
            target.extend(variable)
        else:
            target.append(variable)

    def parse_variable_declaration(self):
        qualifiers = self.parse_declaration_prefixes()
        self.parse_alignment_attributes()
        self.skip_gnu_attributes()

        vtype = self.parse_type()
        self.parse_interleaved_declaration_qualifiers(qualifiers)
        self.parse_alignment_attributes()
        self.skip_gnu_attributes()
        name = self.parse_name_component()
        vtype += self.parse_array_suffix()
        self.skip_bitfield_width()

        value = None
        if self.current_token[0] == "ASSIGN":
            self.eat("ASSIGN")
            value = self.parse_expression()
        elif self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()
            value = FunctionCallNode(self.constructor_initializer_name(vtype), args)
        elif self.current_token[0] == "LBRACE":
            value = self.parse_initializer_list()

        var = VariableNode(vtype, name, value, qualifiers)

        if self.block_depth > 0:
            refuse_local_memory_space(qualifiers, name)
        if "__shared__" in qualifiers:
            return self.create_shared_memory_node(vtype, name, qualifiers)
        elif "__constant__" in qualifiers:
            return ConstantMemoryNode(vtype, name, value, qualifiers)
        else:
            return var

    def parse_variable_declaration_list(self):
        qualifiers = self.parse_declaration_prefixes()
        layout = self.parse_alignment_attributes()
        layout += self.skip_gnu_attributes()

        first_type = self.parse_type()
        base_type = self.strip_declarator_markers(first_type)
        declarations = [
            self.parse_variable_declarator(
                first_type, qualifiers, allow_prefix=False, layout=layout
            )
        ]

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            declarations.append(
                self.parse_variable_declarator(
                    base_type, qualifiers, allow_prefix=True, layout=layout
                )
            )

        return declarations

    def parse_declaration_prefixes(self):
        qualifiers = []
        while self.current_token[0] != "EOF":
            if self.is_cuda_declaration_attribute_identifier():
                self.parse_cuda_declaration_attributes()
                continue

            token_type = self.current_token[0]
            if token_type in self.DECLARATION_PREFIX_TOKENS:
                qualifiers.append(self.current_token[1])
                self.eat(token_type)
                continue
            if (
                token_type == "VOLATILE"
                and self.current_index + 1 < len(self.tokens)
                and self.tokens[self.current_index + 1][0]
                in self.CUDA_STORAGE_QUALIFIER_TOKENS
            ):
                qualifiers.append(self.current_token[1])
                self.eat(token_type)
                continue
            break
        return qualifiers

    def parse_interleaved_declaration_qualifiers(self, qualifiers):
        while self.current_token[0] in self.DECLARATION_PREFIX_TOKENS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

    def parse_variable_declarator(self, base_type, qualifiers, allow_prefix, layout=()):
        vtype = base_type
        if allow_prefix:
            vtype = self.parse_declarator_prefix(vtype)

        declarator_qualifiers = list(qualifiers)
        self.parse_interleaved_declaration_qualifiers(declarator_qualifiers)
        layout = [*layout, *self.parse_alignment_attributes()]
        name = self.parse_name_component()
        vtype += self.parse_array_suffix()
        if self.skip_bitfield_width():
            layout.append("a bit-field width")
        value = self.parse_variable_initializer(vtype)

        if self.block_depth > 0:
            refuse_local_memory_space(declarator_qualifiers, name)
        if "__shared__" in declarator_qualifiers:
            node = self.create_shared_memory_node(vtype, name, declarator_qualifiers)
        elif "__constant__" in declarator_qualifiers:
            node = ConstantMemoryNode(vtype, name, value, declarator_qualifiers)
        else:
            node = VariableNode(vtype, name, value, declarator_qualifiers)
        return self.mark_declared_layout(node, layout)

    @staticmethod
    def mark_declared_layout(node, layout):
        """``node`` with ``layout_specifiers`` set to the layout-changing specifiers its declaration had.

        The parser drops alignment specifiers, ``aligned`` and ``packed`` attributes and bit-field widths from
        declarators. An importer that places a record's members, as the union layout contract does, refuses a
        member that had one instead of placing it at its type's natural layout.
        """
        specifiers = tuple(
            text
            for text in layout
            if not str(text).startswith("__attribute__")
            or re.search(r"align|packed", str(text), re.IGNORECASE)
        )
        if specifiers:
            node.layout_specifiers = specifiers
        return node

    def create_shared_memory_node(self, vtype, name, qualifiers):
        is_extern = "extern" in qualifiers
        is_dynamic = (
            is_extern
            and isinstance(vtype, str)
            and self.has_unsized_array_dimension(vtype)
        )
        return SharedMemoryNode(
            vtype,
            name,
            is_extern=is_extern,
            is_dynamic=is_dynamic,
            qualifiers=qualifiers,
        )

    def has_unsized_array_dimension(self, vtype):
        return "[]" in vtype

    def parse_declarator_prefix(self, base_type):
        parts = [base_type] if base_type else []

        while self.current_token[0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            if self.current_token[0] in self.POSTFIX_TYPE_QUALIFIER_TOKENS:
                parts.append(self.current_token[1])
                self.eat(self.current_token[0])
                continue

            parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(parts)

        return " ".join(parts)

    def parse_variable_initializer(self, vtype):
        if self.current_token[0] == "ASSIGN":
            self.eat("ASSIGN")
            return self.parse_expression()
        if self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()
            return FunctionCallNode(self.constructor_initializer_name(vtype), args)
        if self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        return None

    def skip_bitfield_width(self):
        """Skip a ``: width`` bit-field suffix; return whether there was one."""
        if self.current_token[0] != "COLON":
            return False

        self.eat("COLON")
        if self.current_token[0] not in {"COMMA", "SEMICOLON", "EOF"}:
            self.parse_expression()
        return True

    def constructor_initializer_name(self, vtype):
        return " ".join(
            part
            for part in str(vtype).split()
            if part not in {
                "const",
                "volatile",
                "__restrict__",
                "__restrict",
                "restrict",
                "&",
                "&&",
            }
        )

    def parse_function_pointer_variable_declaration(self):
        qualifiers = self.parse_declaration_prefixes()
        self.parse_alignment_attributes()

        return_type = self.parse_type()
        self.eat("LPAREN")
        pointer_prefix = self.parse_declarator_prefix("")
        name = self.parse_name_component()
        self.eat("RPAREN")

        params = []
        if self.current_token[0] == "LPAREN":
            params = self.parse_parameters()

        vtype = f"{return_type} ({pointer_prefix})".strip()
        value = self.parse_variable_initializer(vtype)
        variable = VariableNode(vtype, name, value, qualifiers)
        variable.params = params
        return variable

    def strip_declarator_markers(self, vtype):
        parts = str(vtype).split()
        declarator_markers = {
            "*",
            "&",
            "&&",
            "__restrict__",
            "__restrict",
            "restrict",
        }
        while parts and parts[-1] in declarator_markers:
            parts.pop()
        return " ".join(parts)

    def parse_array_suffix(self):
        suffixes = []

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
                suffixes.append(f"[{self.expression_to_text(size)}]")
            else:
                suffixes.append("[]")
            self.eat("RBRACKET")

        return "".join(suffixes)

    def parse_argument_list(self):
        self.eat("LPAREN")
        args = []
        self.parse_expression_arguments(args, "RPAREN")
        self.eat("RPAREN")
        return args

    def parse_expression_arguments(self, args, terminator):
        while self.current_token[0] != terminator:
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue

            args.append(self.parse_expression())

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != terminator:
                break

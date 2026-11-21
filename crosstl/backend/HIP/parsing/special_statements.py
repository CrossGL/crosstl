"""delete, label, goto, throw and try statements, preprocessor lines, pragma pack and linkage specifications in the HIP parser."""

import re

from ..HipAst import (
    DeleteNode,
    FunctionCallNode,
    PreprocessorNode,
)


class HipSpecialStatementMixin:
    """Special statement members of ``HipParser``."""

    def parse_delete_statement(self):
        self.advance()
        is_array = False

        if self.match("LBRACKET"):
            self.consume("LBRACKET")
            self.consume("RBRACKET")
            is_array = True

        expression = self.parse_unary_expression()
        if self.match("SEMICOLON"):
            self.advance()
        return DeleteNode(expression, is_array)

    def is_label_statement_start(self):
        return self.match("IDENTIFIER") and self.peek() and self.peek().type == "COLON"

    def parse_label_statement(self):
        self.consume("IDENTIFIER")
        self.consume("COLON")
        self.skip_newlines()

        if not self.current_token or self.match("RBRACE"):
            return None
        return self.parse_statement()

    def parse_goto_statement(self):
        self.consume("GOTO")

        if self.is_declarator_name_token():
            self.advance()
        else:
            while self.current_token and not self.match("SEMICOLON", "RBRACE"):
                self.advance()

        if self.match("SEMICOLON"):
            self.advance()
        return None

    def parse_throw_statement(self):
        self.advance()
        args = []

        if not self.match("SEMICOLON"):
            args.append(self.parse_expression())

        if self.match("SEMICOLON"):
            self.advance()

        return FunctionCallNode("throw", args)

    def parse_try_statement(self):
        self.advance()
        self.skip_newlines()

        statements = []
        if self.match("LBRACE"):
            statements.extend(self.parse_block())
        else:
            stmt = self.parse_statement()
            if stmt:
                statements.extend(stmt if isinstance(stmt, list) else [stmt])

        self.skip_newlines()
        while self.is_identifier_value("catch"):
            self.advance()
            self.skip_newlines()
            if self.match("LPAREN"):
                self.skip_balanced_parentheses()
            self.skip_newlines()
            if self.match("LBRACE"):
                statements.extend(self.parse_block())
            else:
                stmt = self.parse_statement()
                if stmt:
                    statements.extend(stmt if isinstance(stmt, list) else [stmt])
            self.skip_newlines()

        return statements

    def parse_preprocessor(self):
        self.consume("HASH")

        if not self.current_token:
            self.error("Expected preprocessor directive after #")

        directive = self.current_token.value
        self.advance()

        content = []
        while self.current_token and not self.match("NEWLINE"):
            content.append(self.current_token.value)
            self.advance()

        content_text = " ".join(content)
        self.update_pragma_pack_state(directive, content_text)
        return PreprocessorNode(directive, content_text)

    def update_pragma_pack_state(self, directive, content):
        if str(directive).lower() != "pragma":
            return
        compact = re.sub(r"\s+", "", str(content))
        match = re.fullmatch(r"pack\((.*)\)", compact, re.IGNORECASE)
        if match is None:
            return

        arguments = [part for part in match.group(1).split(",") if part]
        if not arguments:
            self.pragma_pack_alignment = None
            return

        command = arguments[0].lower()
        if command == "push":
            self.pragma_pack_stack.append(self.pragma_pack_alignment)
            numeric = next(
                (part for part in reversed(arguments[1:]) if part.isdigit()), None
            )
            if numeric is not None:
                self.pragma_pack_alignment = int(numeric)
            return
        if command == "pop":
            self.pragma_pack_alignment = (
                self.pragma_pack_stack.pop() if self.pragma_pack_stack else None
            )
            return
        if command.isdigit():
            self.pragma_pack_alignment = int(command)

    def current_record_layout_attributes(self, attributes):
        attributes = list(attributes or [])
        if self.pragma_pack_alignment is not None:
            attributes.append(f"pragma_pack({self.pragma_pack_alignment})")
        return attributes

    def is_linkage_specifier_start(self):
        return (
            self.match("EXTERN")
            and self.peek() is not None
            and self.peek().type == "STRING"
        )

    def parse_linkage_specification(self):
        self.consume("EXTERN")
        language = self.consume("STRING").value.strip('"')
        self.skip_newlines()

        if self.match("LBRACE"):
            self.consume("LBRACE")
            statements = []
            while self.current_token and not self.match("RBRACE"):
                if self.match("NEWLINE", "SEMICOLON"):
                    self.advance()
                    continue

                stmt = self.parse_statement()
                if stmt:
                    if isinstance(stmt, list):
                        statements.extend(stmt)
                    else:
                        statements.append(stmt)

            self.consume("RBRACE")
            self.apply_linkage_to_statement(statements, language)
            return statements

        stmt = self.parse_statement()
        self.apply_linkage_to_statement(stmt, language)
        return stmt

    def apply_linkage_to_statement(self, stmt, language):
        if isinstance(stmt, list):
            for item in stmt:
                self.apply_linkage_to_statement(item, language)
            return

        if not stmt:
            return

        if hasattr(stmt, "qualifiers"):
            qualifiers = getattr(stmt, "qualifiers", []) or []
            if "extern" not in qualifiers:
                qualifiers.insert(0, "extern")
            stmt.qualifiers = qualifiers
            stmt.linkage = language

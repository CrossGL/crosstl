"""Control-flow, synchronization, inline assembly, block and expression statements in the HIP parser."""

from ..HipAst import (
    BlockNode,
    CaseNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    HipAsmNode,
    HipAsmOperandNode,
    IfNode,
    RangeForNode,
    ReturnNode,
    SwitchNode,
    SyncNode,
    WhileNode,
)


class HipControlFlowMixin:
    """Control flow members of ``HipParser``."""

    def parse_return_statement(self):
        self.consume("RETURN")

        value = None
        if not self.match("SEMICOLON"):
            value = self.parse_expression()

        if self.match("SEMICOLON"):
            self.advance()

        return ReturnNode(value)

    def parse_if_statement(self):
        self.consume("IF")
        self.skip_newlines()
        if self.match("CONSTEXPR"):
            self.advance()
            self.skip_newlines()
        self.consume("LPAREN")

        condition_or_init = None
        if not self.match("SEMICOLON"):
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list(
                    consume_semicolon=False
                )
                condition_or_init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
                )
            else:
                condition_or_init = self.parse_expression()

        init = None
        if self.match("SEMICOLON"):
            self.advance()
            init = condition_or_init
            self.skip_newlines()
            condition = self.parse_expression()
        else:
            condition = condition_or_init

        self.consume("RPAREN")

        if_body = None
        self.skip_newlines()
        if self.match("LBRACE"):
            if_body = self.parse_block()
        else:
            if_body = self.parse_statement()

        else_body = None
        self.skip_newlines()
        if self.match("ELSE"):
            self.advance()
            self.skip_newlines()
            if self.match("LBRACE"):
                else_body = self.parse_block()
            else:
                else_body = self.parse_statement()

        statement = IfNode(condition, if_body, else_body)
        if init is not None:
            return [*init, statement] if isinstance(init, list) else [init, statement]
        return statement

    def parse_for_statement(self):
        self.consume("FOR")
        self.consume("LPAREN")

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
        self.consume("SEMICOLON")
        self.skip_newlines()

        condition = None
        if not self.match("SEMICOLON"):
            condition = self.parse_expression()
        self.consume("SEMICOLON")
        self.skip_newlines()

        update = None
        if not self.match("RPAREN"):
            update = self.parse_for_update_expression()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return ForNode(init, condition, update, body)

    def parse_for_update_expression(self):
        updates = [self.parse_expression()]
        while self.match("COMMA"):
            self.advance()
            updates.append(self.parse_expression())

        return updates if len(updates) > 1 else updates[0]

    def is_range_for_statement(self):
        index = self.skip_range_for_type_at_pos(self.pos)
        if index is None:
            return False

        if index >= len(self.tokens) or self.tokens[index].type != "IDENTIFIER":
            return False

        index += 1
        index = self.skip_array_suffix_at_pos(index)
        return index < len(self.tokens) and self.tokens[index].type == "COLON"

    def skip_range_for_type_at_pos(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS
        ):
            index += 1

        if index >= len(self.tokens) or not self.is_type_token(self.tokens[index]):
            return None

        index += 1
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "SCOPE"
            and self.tokens[index + 1].type == "IDENTIFIER"
        ):
            index += 2

        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        index = self.skip_postfix_type_qualifiers_at_pos(index)
        while index < len(self.tokens) and self.tokens[index].type in {
            "ASTERISK",
            "STAR",
            *self.TYPE_REFERENCE_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_pos(index)

        return self.skip_array_suffix_at_pos(index)

    def parse_range_for_statement(self):
        vtype = self.parse_type()
        name = self.consume("IDENTIFIER").value
        self.consume("COLON")
        self.skip_newlines()
        iterable = self.parse_expression()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return RangeForNode(vtype, name, iterable, body)

    def parse_while_statement(self):
        self.consume("WHILE")
        self.consume("LPAREN")
        condition = self.parse_expression()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.consume("DO")
        self.skip_newlines()
        body = self.parse_statement()
        self.skip_newlines()
        self.consume("WHILE")
        self.consume("LPAREN")
        condition = self.parse_expression()
        self.consume("RPAREN")

        if self.match("SEMICOLON"):
            self.advance()

        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.consume("SWITCH")
        self.consume("LPAREN")
        expression = self.parse_expression()
        self.consume("RPAREN")
        self.skip_newlines()
        self.consume("LBRACE")

        cases = []
        ordered_cases = []
        default_case = None
        seen_default = False

        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            if self.match("CASE"):
                self.advance()
                value = self.parse_expression()
                self.consume("COLON")
                body = []
                while self.current_token and not self.match(
                    "CASE", "DEFAULT", "RBRACE"
                ):
                    stmt = self.parse_scoped_statement()
                    if stmt:
                        body.append(stmt)
                case = CaseNode(value, body)
                cases.append(case)
                ordered_cases.append(case)
            elif self.match("DEFAULT"):
                if seen_default:
                    raise self.syntax_error("duplicate default label in switch")
                seen_default = True
                self.advance()
                self.consume("COLON")
                default_case = []
                while self.current_token and not self.match("CASE", "RBRACE"):
                    stmt = self.parse_scoped_statement()
                    if stmt:
                        default_case.append(stmt)
                ordered_cases.append(CaseNode(None, default_case))
            else:
                self.advance()

        self.consume("RBRACE")
        switch = SwitchNode(expression, cases, default_case)
        switch.ordered_cases = ordered_cases
        return switch

    def parse_sync_statement(self):
        sync_type = self.current_token.value
        self.advance()

        self.consume("LPAREN")
        args = []
        if not self.match("RPAREN"):
            args.append(self.parse_expression())
            while self.match("COMMA"):
                self.advance()
                args.append(self.parse_expression())
        self.consume("RPAREN")

        if self.match("SEMICOLON"):
            self.advance()

        return SyncNode(sync_type, args)

    def parse_asm_statement(self):
        self.consume("ASM")
        is_volatile = False
        if self.match("VOLATILE"):
            is_volatile = True
            self.advance()

        self.consume("LPAREN")
        self.skip_newlines()
        template = self.parse_asm_string_literal()
        outputs = []
        inputs = []
        clobbers = []

        self.skip_newlines()
        section_index = self.consume_asm_section_delimiters()
        if section_index == 1:
            outputs = self.parse_asm_operands()
        elif section_index == 2:
            inputs = self.parse_asm_operands()
        elif section_index >= 3:
            clobbers = self.parse_asm_clobbers()

        self.skip_newlines()
        while self.match("COLON", "SCOPE"):
            section_index += self.consume_asm_section_delimiters()
            if section_index == 2:
                inputs = self.parse_asm_operands()
            elif section_index >= 3:
                clobbers = self.parse_asm_clobbers()
                break
            self.skip_newlines()

        self.skip_newlines()
        self.consume("RPAREN")
        if self.match("SEMICOLON"):
            self.advance()
        return HipAsmNode(template, outputs, inputs, clobbers, is_volatile)

    def consume_asm_section_delimiters(self):
        delimiter_count = 0
        self.skip_newlines()
        while self.match("COLON", "SCOPE"):
            if self.match("COLON"):
                delimiter_count += 1
                self.advance()
            else:
                delimiter_count += 2
                self.advance()
            self.skip_newlines()
        return delimiter_count

    def parse_asm_string_literal(self):
        value = self.consume("STRING").value
        self.skip_newlines()
        while self.match("STRING"):
            next_value = self.consume("STRING").value
            if value.endswith('"') and next_value.startswith('"'):
                value = value[:-1] + next_value[1:]
            else:
                value += next_value
            self.skip_newlines()
        return value

    def parse_asm_operands(self):
        operands = []

        self.skip_newlines()
        while self.current_token and not self.match("COLON", "SCOPE", "RPAREN"):
            symbolic_name = None
            if self.match("LBRACKET"):
                self.consume("LBRACKET")
                symbolic_name = self.consume("IDENTIFIER").value
                self.consume("RBRACKET")

            constraint = self.consume("STRING").value
            expression = None
            if self.match("LPAREN"):
                self.consume("LPAREN")
                if not self.match("RPAREN"):
                    expression = self.parse_expression()
                self.consume("RPAREN")

            operands.append(HipAsmOperandNode(constraint, expression, symbolic_name))
            if not self.match("COMMA"):
                break
            self.advance()
            self.skip_newlines()

        return operands

    def parse_asm_clobbers(self):
        clobbers = []

        self.skip_newlines()
        while self.current_token and not self.match("RPAREN"):
            clobbers.append(self.consume("STRING").value)
            if not self.match("COMMA"):
                break
            self.advance()
            self.skip_newlines()

        return clobbers

    def parse_scoped_statement(self):
        """A statement of a block or a ``case``; a nested ``{ ... }`` keeps its own scope as a ``BlockNode``."""
        if self.match("LBRACE"):
            return BlockNode(self.parse_block())
        return self.parse_statement()

    def parse_block(self):
        self.consume("LBRACE")
        statements = []

        self.block_depth += 1
        try:
            while self.current_token and not self.match("RBRACE"):
                stmt = self.parse_scoped_statement()
                if stmt:
                    if isinstance(stmt, list):
                        statements.extend(stmt)
                    else:
                        statements.append(stmt)
        finally:
            self.block_depth -= 1

        self.consume("RBRACE")
        return statements

    def parse_expression_statement(self):
        expr = self.parse_expression()

        if self.match("SEMICOLON"):
            self.advance()
            return expr
        if self.is_optional_semicolon_macro_statement(expr):
            self.skip_newlines()
            if self.match("SEMICOLON"):
                self.advance()
            return expr

        self.skip_newlines()
        self.consume("SEMICOLON")

        return expr

    def is_optional_semicolon_macro_statement(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return False
        name = expr.name
        is_macro_call = (
            isinstance(name, str)
            and (name.isupper() or name in self.OPTIONAL_SEMICOLON_MACRO_NAMES)
            and any(char.isalpha() for char in name)
        )
        if not is_macro_call:
            return False
        if self.is_expression_statement_boundary():
            return True
        return name in self.OPTIONAL_SEMICOLON_MACRO_NAMES and (
            self.match("IDENTIFIER")
            and self.current_token.value in self.OPTIONAL_SEMICOLON_MACRO_NAMES
        )

    def is_expression_statement_boundary(self):
        if not self.current_token:
            return True
        if self.match("NEWLINE", "RBRACE"):
            return True
        if self.pos > 0 and self.tokens[self.pos - 1].type == "NEWLINE":
            return True
        return False

"""Control-flow, return, synchronization and inline assembly statements in the CUDA parser."""

from crosstl.translator.refusals import CudaSyntaxError

from ..CudaAst import (
    CaseNode,
    CudaAsmNode,
    CudaAsmOperandNode,
    DoWhileNode,
    ForNode,
    IfNode,
    RangeForNode,
    ReturnNode,
    SwitchNode,
    SyncNode,
    WhileNode,
)


class CudaControlFlowMixin:
    """Control flow members of ``CudaParser``."""

    def is_variable_declaration(self):
        saved_index = self.current_index

        saved_index = self.skip_declaration_prefixes_at_index(saved_index)
        saved_index = self.skip_alignment_attributes_at_index(saved_index)
        saved_index = self.skip_gnu_attributes_at_index(saved_index)

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )

        if saved_index is not None:
            saved_index = self.skip_interleaved_declaration_qualifiers_at_index(
                saved_index
            )
            saved_index = self.skip_alignment_attributes_at_index(saved_index)
            saved_index = self.skip_gnu_attributes_at_index(saved_index)
            if (
                saved_index < len(self.tokens)
                and self.tokens[saved_index][0] in self.NAME_COMPONENT_TOKENS
            ):
                saved_index += 1
                if saved_index >= len(self.tokens):
                    return True
                return self.tokens[saved_index][0] in [
                    "SEMICOLON",
                    "ASSIGN",
                    "LBRACKET",
                    "LPAREN",
                    "LBRACE",
                    "COMMA",
                ]
        return False

    def parse_if_statement(self):
        self.eat("IF")
        is_constexpr = False
        if self.current_token[0] == "CONSTEXPR":
            is_constexpr = True
            self.eat("CONSTEXPR")
        self.eat("LPAREN")

        condition_or_init = None
        if self.current_token[0] != "SEMICOLON":
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list()
                condition_or_init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
                )
            else:
                condition_or_init = self.parse_expression()

        init = None
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            init = condition_or_init
            condition = self.parse_expression()
        else:
            condition = condition_or_init

        self.eat("RPAREN")

        if_body = self.parse_statement()

        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement()

        node = IfNode(condition, if_body, else_body)
        node.is_constexpr = is_constexpr
        if init is not None:
            return [*init, node] if isinstance(init, list) else [init, node]
        return node

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")

        if self.is_range_for_statement():
            return self.parse_range_for_statement()

        init = None
        if self.current_token[0] != "SEMICOLON":
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list()
                init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
                )
            else:
                init = self.parse_for_update_expression()
        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_for_update_expression()
        self.eat("RPAREN")

        body = self.parse_statement()

        return ForNode(init, condition, update, body)

    def parse_for_update_expression(self):
        updates = [self.parse_assignment_expression()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            updates.append(self.parse_assignment_expression())
        return updates if len(updates) > 1 else updates[0]

    def is_range_for_statement(self):
        index = self.skip_range_for_type_at_index(self.current_index)
        if index is None:
            return False

        if index >= len(self.tokens) or self.tokens[index][0] != "IDENTIFIER":
            return False

        index += 1
        index = self.skip_array_suffix_at_index(index)
        return index < len(self.tokens) and self.tokens[index][0] == "COLON"

    def skip_range_for_type_at_index(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_QUALIFIER_TOKENS
        ):
            index += 1

        if index >= len(self.tokens) or self.tokens[index][0] not in self.TYPE_TOKENS:
            return None

        type_token = self.tokens[index][0]
        index += 1
        index = self.skip_composite_scalar_type_suffix_at_index(index, type_token)
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "SCOPE"
            and self.tokens[index + 1][0] == "IDENTIFIER"
        ):
            index += 2

        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            index = self.skip_template_at_index(index)
            if index is None:
                return None

        index = self.skip_postfix_type_qualifiers_at_index(index)
        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        return self.skip_array_suffix_at_index(index)

    def parse_range_for_statement(self):
        vtype = self.parse_type()
        name = self.eat("IDENTIFIER")[1]
        self.eat("COLON")
        iterable = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement()

        return RangeForNode(vtype, name, iterable, body)

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement()

        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        body = self.parse_statement()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")

        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        ordered_cases = []
        default_case = None
        seen_default = False

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "CASE":
                self.eat("CASE")
                value = self.parse_expression()
                self.eat("COLON")
                body = []
                while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE"]:
                    body.append(self.parse_scoped_statement())
                case = CaseNode(value, body)
                cases.append(case)
                ordered_cases.append(case)
            elif self.current_token[0] == "DEFAULT":
                if seen_default:
                    raise CudaSyntaxError("duplicate default label in switch")
                seen_default = True
                self.eat("DEFAULT")
                self.eat("COLON")
                default_case = []
                while self.current_token[0] not in ["CASE", "RBRACE"]:
                    default_case.append(self.parse_scoped_statement())
                ordered_cases.append(CaseNode(None, default_case))
            else:
                self.eat(self.current_token[0])

        self.eat("RBRACE")
        switch = SwitchNode(expression, cases, default_case)
        switch.ordered_cases = ordered_cases
        return switch

    def parse_return_statement(self):
        self.eat("RETURN")

        value = None
        if self.current_token[0] != "SEMICOLON":
            value = self.parse_expression()

        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_sync_statement(self):
        sync_type = self.current_token[1]
        self.eat(self.current_token[0])

        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        self.eat("SEMICOLON")

        return SyncNode(sync_type, args)

    def parse_asm_statement(self):
        self.eat("ASM")
        is_volatile = False
        if self.current_token[0] == "VOLATILE":
            is_volatile = True
            self.eat("VOLATILE")

        self.eat("LPAREN")
        template = self.parse_asm_string_literal()
        outputs = []
        inputs = []
        clobbers = []

        section_index = self.consume_asm_section_delimiters()
        if section_index == 1:
            outputs = self.parse_asm_operands()
        elif section_index == 2:
            inputs = self.parse_asm_operands()
        elif section_index >= 3:
            clobbers = self.parse_asm_clobbers()

        while self.current_token[0] in {"COLON", "SCOPE"}:
            section_index += self.consume_asm_section_delimiters()
            if section_index == 2:
                inputs = self.parse_asm_operands()
            elif section_index >= 3:
                clobbers = self.parse_asm_clobbers()
                break

        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return CudaAsmNode(template, outputs, inputs, clobbers, is_volatile)

    def consume_asm_section_delimiters(self):
        delimiter_count = 0
        while self.current_token[0] in {"COLON", "SCOPE"}:
            if self.current_token[0] == "COLON":
                delimiter_count += 1
                self.eat("COLON")
            else:
                delimiter_count += 2
                self.eat("SCOPE")
        return delimiter_count

    def parse_asm_string_literal(self):
        value = self.eat("STRING")[1]
        while self.current_token[0] == "STRING":
            next_value = self.eat("STRING")[1]
            if value.endswith('"') and next_value.startswith('"'):
                value = value[:-1] + next_value[1:]
            else:
                value += next_value
        return value

    def parse_asm_operands(self):
        operands = []

        while self.current_token[0] not in {"COLON", "SCOPE", "RPAREN", "EOF"}:
            symbolic_name = None
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                symbolic_name = self.eat("IDENTIFIER")[1]
                self.eat("RBRACKET")

            constraint = self.eat("STRING")[1]
            expression = None
            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                if self.current_token[0] != "RPAREN":
                    expression = self.parse_expression()
                self.eat("RPAREN")

            operands.append(CudaAsmOperandNode(constraint, expression, symbolic_name))
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        return operands

    def parse_asm_clobbers(self):
        clobbers = []

        while self.current_token[0] not in {"RPAREN", "EOF"}:
            clobbers.append(self.eat("STRING")[1])
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        return clobbers

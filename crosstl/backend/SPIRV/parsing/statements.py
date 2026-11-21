"""Statements in the GLSL-style text form accepted by the SPIR-V parser: blocks, returns, updates, if, for, variables, while, do-while, switch, case, default and uniform declarations."""

from crosstl.translator.refusals import SpirvSyntaxError

from ..VulkanAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DefaultNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    UniformNode,
    VariableNode,
    WhileNode,
)
from ..VulkanLexer import VALID_DATA_TYPES


class VulkanStatementMixin:
    """Statement members of ``VulkanParser``."""

    def parse_block(self):
        self.eat("LBRACE")
        statements = []
        while self.current_token[0] != "RBRACE":
            statement = self.parse_body()
            if isinstance(statement, list):
                statements.extend(statement)
            elif statement is not None:
                statements.append(statement)
        self.eat("RBRACE")
        return statements

    def parse_statement_or_block(self):
        if self.current_token[0] == "LBRACE":
            return self.parse_block()
        statement = self.parse_body()
        if isinstance(statement, list):
            return statement
        if statement is None:
            return []
        return [statement]

    def parse_body(self):
        token_type = self.current_token[0]

        if token_type == "SEMICOLON":
            self.eat("SEMICOLON")
            return []
        if token_type == "LBRACE":
            return self.parse_block()
        if token_type == "CONST":
            return self.parse_assignment_or_function_call()
        if self.current_token[1] in self.DECLARATION_QUALIFIERS:
            return self.parse_assignment_or_function_call()
        if token_type == "IDENTIFIER" and (
            self.peek(1) in ["LPAREN", "LBRACKET"]
            or self.looks_like_member_call_statement()
        ):
            return self.parse_expression_statement()
        if token_type == "IDENTIFIER" or self.current_token[1] in VALID_DATA_TYPES:
            return self.parse_assignment_or_function_call()
        elif token_type == "IF":
            return self.parse_if_statement()
        elif token_type == "FOR":
            return self.parse_for_statement()
        elif token_type == "WHILE":
            return self.parse_while_statement()
        elif token_type == "DO":
            return self.parse_do_while_statement()
        elif token_type == "SWITCH":
            return self.parse_switch_statement()
        elif token_type == "BREAK":
            if self.breakable_depth == 0:
                raise SpirvSyntaxError("break used outside loop or switch")
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        elif token_type == "CONTINUE":
            if self.loop_depth == 0:
                raise SpirvSyntaxError("continue used outside loop")
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return ContinueNode()
        elif token_type == "RETURN":
            return self.parse_return_statement()
        elif token_type == "DISCARD":
            self.eat("DISCARD")
            self.eat("SEMICOLON")
            return DiscardNode()
        else:
            return self.parse_expression_statement()

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode()

        value = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_update(self):
        if self.current_token[0] == "IDENTIFIER":
            target = self.parse_update_target()
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                return UnaryOpNode("POST_INCREMENT", target)
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                return UnaryOpNode("POST_DECREMENT", target)
            elif self.current_token[0] in self.ASSIGNMENT_TOKENS:
                op_name = self.current_token[1]
                self.eat(self.current_token[0])
                value = self.parse_expression()
                return AssignmentNode(target, value, op_name)
            else:
                raise SpirvSyntaxError(
                    f"Unexpected token in update: {self.current_token[0]}"
                )
        elif self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_update_target())
        elif self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_update_target())
        else:
            raise SpirvSyntaxError(
                f"Unexpected token in update: {self.current_token[0]}"
            )

    def parse_update_target(self):
        if self.current_token[0] != "IDENTIFIER":
            raise SpirvSyntaxError(
                f"Expected update target, got {self.current_token[0]}"
            )

        target = VariableNode("", self.current_token[1])
        self.eat("IDENTIFIER")
        target = self.parse_postfix_suffixes(target)
        if not isinstance(target, (VariableNode, MemberAccessNode, ArrayAccessNode)):
            raise SpirvSyntaxError(f"Invalid update target: {type(target).__name__}")
        return target

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        if_condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_statement_or_block()
        else_body = None
        else_if_chain = []
        while self.current_token[0] == "ELSE" and self.peek(1) == "IF":
            self.eat("ELSE")
            self.eat("IF")
            self.eat("LPAREN")
            else_if_condition = self.parse_expression()
            self.eat("RPAREN")
            else_if_chain.append((else_if_condition, self.parse_statement_or_block()))
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement_or_block()
        return IfNode(
            if_condition,
            if_body,
            else_body,
            else_if_chain=else_if_chain,
        )

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")
        initialization = self.parse_for_initializer()
        condition = self.parse_for_condition()
        increment = self.parse_for_update()
        self.eat("RPAREN")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_statement_or_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        return ForNode(initialization, condition, increment, body)

    def parse_for_initializer(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        if self.current_token[0] in {"PRE_INCREMENT", "PRE_DECREMENT"}:
            item = self.parse_update()
            self.eat("SEMICOLON")
            return item

        items = [
            self.parse_assignment_or_function_call(
                terminators={"COMMA", "SEMICOLON"},
                consume_terminator=False,
            )
        ]
        declaration_type = self.for_initializer_declaration_type(items[0])
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if declaration_type and self.current_token[0] == "IDENTIFIER":
                items.append(
                    self.parse_variable(
                        declaration_type,
                        terminators={"COMMA", "SEMICOLON"},
                        consume_terminator=False,
                    )
                )
            else:
                items.append(
                    self.parse_assignment_or_function_call(
                        terminators={"COMMA", "SEMICOLON"},
                        consume_terminator=False,
                    )
                )
        self.eat("SEMICOLON")
        return items if len(items) > 1 else items[0]

    def for_initializer_declaration_type(self, item):
        target = item.left if isinstance(item, AssignmentNode) else item
        if isinstance(target, VariableNode) and target.vtype:
            return target.vtype
        return ""

    def parse_for_condition(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        condition = self.parse_expression()
        self.eat("SEMICOLON")
        return condition

    def parse_for_update(self):
        if self.current_token[0] == "RPAREN":
            return None
        updates = [self.parse_update()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            updates.append(self.parse_update())
        return updates if len(updates) > 1 else updates[0]

    def parse_variable(
        self,
        type_name="",
        terminators=None,
        consume_terminator=True,
        expected_terminators=None,
    ):
        terminators = terminators or {"SEMICOLON"}
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if type_name:
            name += self.parse_array_suffixes_as_text()
        target = VariableNode(type_name, name)
        if not type_name:
            target = self.parse_postfix_suffixes(target)
        else:
            target.semantic = self.parse_optional_hlsl_semantic()

        if type_name in {"cbuffer", "tbuffer"} and self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_tokens()
            if consume_terminator and self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return target

        if self.current_token[0] in terminators:
            if consume_terminator:
                self.eat(self.current_token[0])
            return target

        elif self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op_name = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()
            self.consume_terminator(
                terminators,
                consume_terminator,
                expected_terminators=expected_terminators,
            )
            return AssignmentNode(target, value, op_name)

        elif self.current_token[0] in ("BINARY_AND", "BINARY_OR", "BINARY_XOR"):
            op = self.current_token[0]
            op_symbol = (
                "&" if op == "BINARY_AND" else ("|" if op == "BINARY_OR" else "^")
            )
            self.eat(op)
            right = self.parse_expression()
            self.consume_terminator(
                terminators,
                consume_terminator,
                expected_terminators=expected_terminators,
            )
            return BinaryOpNode(target, op_symbol, right)

        elif self.current_token[0] in (
            "EQUAL",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
        ):
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)
            value = self.parse_expression()
            self.consume_terminator(
                terminators,
                consume_terminator,
                expected_terminators=expected_terminators,
            )
            return BinaryOpNode(target, op_name, value)
        else:
            raise SpirvSyntaxError(
                f"Unexpected token after identifier {name}: {self.current_token[0]}"
            )

    def parse_variable_declaration(
        self,
        type_name,
        terminators=None,
        consume_terminator=True,
    ):
        terminators = terminators or {"SEMICOLON"}
        declarator_terminators = {*terminators, "COMMA"}
        declarations = [
            self.parse_variable(
                type_name,
                terminators=declarator_terminators,
                consume_terminator=False,
                expected_terminators=terminators,
            )
        ]
        if (
            isinstance(declarations[0], VariableNode)
            and declarations[0].vtype in {"cbuffer", "tbuffer"}
            and self.current_token[0] not in declarator_terminators
        ):
            return declarations[0]

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            declarations.append(
                self.parse_variable(
                    type_name,
                    terminators=declarator_terminators,
                    consume_terminator=False,
                    expected_terminators=terminators,
                )
            )

        self.consume_terminator(terminators, consume_terminator)
        return declarations if len(declarations) > 1 else declarations[0]

    def consume_terminator(
        self,
        terminators,
        consume_terminator,
        expected_terminators=None,
    ):
        if self.current_token[0] not in terminators:
            expected = " or ".join(sorted(expected_terminators or terminators))
            raise SpirvSyntaxError(f"Expected {expected}, got {self.current_token[0]}")
        if consume_terminator:
            self.eat(self.current_token[0])

    def parse_expression_statement(self):
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_statement_or_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expr = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        cases = []
        seen_default = False
        self.breakable_depth += 1
        try:
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                if self.current_token[0] == "DEFAULT":
                    if seen_default:
                        raise SpirvSyntaxError("duplicate default label in switch")
                    seen_default = True
                cases.append(self.parse_case_statement())
            if self.current_token[0] == "EOF":
                raise SpirvSyntaxError("Unterminated switch statement")
        finally:
            self.breakable_depth -= 1
        self.eat("RBRACE")
        return SwitchNode(expr, cases)

    def parse_case_statement(self):
        if self.current_token[0] == "CASE":
            self.eat("CASE")
            value = self.parse_expression()
            self.eat("COLON")
        elif self.current_token[0] == "DEFAULT":
            self.eat("DEFAULT")
            value = None
            self.eat("COLON")
        else:
            raise SpirvSyntaxError(
                f"Expected CASE or DEFAULT in switch, got {self.current_token[0]}"
            )
        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            statement = self.parse_body()
            if isinstance(statement, list):
                statements.extend(statement)
            elif statement is not None:
                statements.append(statement)
        if self.current_token[0] == "EOF":
            raise SpirvSyntaxError("Unterminated switch case")
        return CaseNode(value, statements)

    def parse_default_statement(self):
        self.eat("DEFAULT")
        self.eat("COLON")
        statements = []
        while self.current_token[0] not in ["CASE", "RBRACE"]:
            statements.append(self.parse_body())
        return DefaultNode(statements)

    def parse_uniform(self):
        self.eat("UNIFORM")
        var_type = self.parse_data_type(allow_identifier=True)
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        name += self.parse_array_suffixes_as_text()
        self.eat("SEMICOLON")
        return UniformNode(var_type, name)

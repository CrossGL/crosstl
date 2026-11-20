"""Statements in the CrossGL parser: blocks, local declarations, control flow, match arms and patterns, returns and switch statements."""

from ..ast import (
    BlockNode,
    BreakNode,
    CaseNode,
    CommaExpressionNode,
    ConstructorPatternNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    IdentifierPatternNode,
    IfNode,
    LiteralPatternNode,
    LoopNode,
    MatchArmNode,
    MatchNode,
    NamedType,
    ReturnNode,
    StructNode,
    StructPatternNode,
    SwitchNode,
    VariableNode,
    WhileNode,
    WildcardPatternNode,
)
from ..refusals import CrossGLSyntaxError
from ..static_assertion_syntax import (
    parse_static_assertion,
    starts_static_assertion,
)


class ParserStatementMixin:
    """Statement members of ``Parser``."""

    def parse_block(self):
        """Parse a braced statement block."""
        self.eat("LBRACE")
        statements = []
        self.enter_value_type_scope()
        try:
            while self.current_token[0] != "RBRACE":
                stmt = self.parse_statement()
                self.append_parsed_nodes(statements, stmt)

            self.eat("RBRACE")
        finally:
            self.restore_value_type_scope()
        return BlockNode(statements)

    def parse_statement(self):
        """Parse any statement form supported by CrossGL."""
        if self.current_token_starts_statement_attribute():
            return self.parse_attributed_statement()
        if self.current_token[0] == "SEMICOLON":
            # A null statement does nothing (GLSL 4.60 section 6.1, C++ [stmt.expr]/1): it adds no node, and a
            # braceless body made of one becomes an empty block through parse_substatement.
            self.eat("SEMICOLON")
            return []
        if starts_static_assertion(self):
            return parse_static_assertion(self)
        if self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
        elif self.current_token[0] == "DO":
            return self.parse_do_while_statement()
        elif self.current_token[0] == "LOOP":
            return self.parse_loop_statement()
        elif self.current_token[0] == "MATCH":
            return self.parse_match_statement()
        elif self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()
        elif self.current_token[0] == "RETURN":
            return self.parse_return_statement()
        elif self.current_token[0] == "LET":
            return self.parse_let_declaration()
        elif self.current_token[0] == "BREAK":
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        elif self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return ContinueNode()
        elif self.current_token[0] == "LBRACE":
            return self.parse_block()
        elif self.current_token[0] == "STRUCT" and self.peek()[0] == "IDENTIFIER":
            # A structure declared in a block; semantics.local_structs hoists it to module scope.
            return self.parse_local_struct_statement()
        elif self.current_token_starts_inferred_var_declaration():
            return self.parse_inferred_var_declaration()
        elif self.is_variable_declaration():
            return self.parse_variable_declaration()
        else:
            expr = self.parse_expression()
            if self.current_token[0] == "COMMA":
                # ``a(), b();``: the comma operator at statement level, the same node as ``(a(), b());``.
                operands = [expr]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    operands.append(self.parse_expression())
                expr = CommaExpressionNode(operands)
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return ExpressionStatementNode(expr)
            if self.current_token[0] == "RBRACE":
                return ExpressionStatementNode(expr, is_tail_expression=True)
            self.eat("SEMICOLON")
            return ExpressionStatementNode(expr)

    def parse_local_struct_statement(self):
        """Parse ``struct Name { ... } [declarators];`` in a block, refusing one it cannot read.

        Declarators after the closing brace (``struct S { float x; } a, b[2];``) declare variables of the
        structure, as in C and GLSL, and follow it in the block.
        """
        node = self.parse_struct()
        if not isinstance(node, StructNode):
            raise CrossGLSyntaxError(
                "CrossGL could not parse the structure declared in a block, at "
                f"{self.current_token[0]} {self.current_token[1]!r}"
            )
        node.is_local_declaration = True
        if self.peek(-1)[0] == "SEMICOLON" or not (
            self.current_token[0] == "IDENTIFIER"
            and self.peek()[0] in {"SEMICOLON", "COMMA", "LBRACKET", "EQUALS"}
        ):
            return node
        declarations = []
        while True:
            declarations.append(
                self.parse_variable_declarator(NamedType(node.name), [], [])
            )
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
        self.eat("SEMICOLON")
        return [node, *declarations]

    def current_token_starts_statement_attribute(self):
        """Return whether the current token starts statement-level metadata."""
        return (
            self.current_token[0] in {"AT", "ATTRIBUTE"}
            or self.current_token_starts_square_attribute()
        )

    def parse_attributed_statement(self):
        """Parse metadata that decorates the following statement."""
        attributes = self.parse_attribute_annotations()
        statement = self.parse_statement()
        if isinstance(statement, list):
            for node in statement:
                existing = getattr(node, "attributes", [])
                node.attributes = [*attributes, *existing]
            return statement
        existing = getattr(statement, "attributes", [])
        statement.attributes = [*attributes, *existing]
        return statement

    def parse_let_declaration(self, attributes=None):
        """Parse Rust-style ``[@attr] let [mut] name [: type] = expr;`` declarations."""
        self.eat("LET")

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        name = self.parse_binding_identifier()
        generic_params = self.parse_declaration_square_generic_parameters()

        var_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            var_type = self.parse_type()

        self.eat("EQUALS")
        initial_value = self.parse_expression()
        self.eat("SEMICOLON")

        var_node = VariableNode(
            name=name,
            var_type=var_type,
            initial_value=initial_value,
            attributes=list(attributes or []),
        )
        var_node.generic_params = generic_params

        if is_mutable:
            var_node.qualifiers = getattr(var_node, "qualifiers", []) + ["mut"]

        var_node.is_let_declaration = True

        return self.finish_variable_contract(var_node)

    def current_token_starts_inferred_var_declaration(self):
        """Return whether ``var name = expr;``, or the refused ``var name;``, starts here."""
        if self.current_token[0] != "VAR":
            return False
        name_value = self.peek()[1]
        return (
            isinstance(name_value, str)
            and name_value.isidentifier()
            and self.peek(2)[0] in {"EQUALS", "SEMICOLON"}
        )

    def parse_inferred_var_declaration(self):
        """Parse ``var name = expr;``: a mutable local typed by its initializer.

        It declares the same variable as ``let mut name = expr;`` (WGSL 7.3 ``var`` declarations and Mojo ``var``
        locals both take the initializer's type when none is written). ``var name;`` writes neither a type nor an
        initializer to take one from, so it is refused rather than given a guessed type.
        """
        self.eat("VAR")
        name = self.parse_binding_identifier()
        if self.current_token[0] != "EQUALS":
            raise CrossGLSyntaxError(
                f"'var {name};' has no type and no initializer to infer one from; "
                f"write 'var {name}: <type>;' or 'var {name} = <value>;'"
            )
        self.eat("EQUALS")
        initial_value = self.parse_expression()
        self.eat("SEMICOLON")

        var_node = VariableNode(name=name, var_type=None, initial_value=initial_value)
        var_node.generic_params = []
        var_node.qualifiers = getattr(var_node, "qualifiers", []) + ["mut"]
        var_node.is_let_declaration = True
        return self.finish_variable_contract(var_node)

    def parse_binding_identifier(self):
        """Parse an identifier in binding position, including keyword-like names."""
        token_type, token_value = self.current_token
        if not self.current_token_is_binding_identifier():
            raise CrossGLSyntaxError(f"Expected binding name, got {token_type}")

        self.eat(token_type)
        return token_value

    def current_token_is_binding_identifier(self):
        token_value = self.current_token[1]
        return isinstance(token_value, str) and token_value.isidentifier()

    def current_token_is_declaration_binding_identifier(self):
        token_type, token_value = self.current_token
        if token_type == "IDENTIFIER":
            return True
        if not (isinstance(token_value, str) and token_value.isidentifier()):
            return False
        return self.peek()[0] in {
            "AT",
            "ATTRIBUTE",
            "COMMA",
            "EQUALS",
            "LBRACKET",
            "RPAREN",
            "SEMICOLON",
        }

    def parse_substatement(self):
        """Parse the body of a selection or iteration statement as a block.

        A substatement that is not a compound statement behaves as if it were one that contains it (C++
        [stmt.select.general]/2 and [stmt.iter.general]/2; GLSL 4.60 sections 6.3-6.4 scope it the same way), so a
        braceless body is kept as a one-statement block and every consumer sees a single shape.
        """
        statement = self.parse_statement()
        if isinstance(statement, BlockNode):
            return statement
        statements = list(statement) if isinstance(statement, list) else [statement]
        return BlockNode(
            statements,
            source_location=(
                getattr(statements[0], "source_location", None) if statements else None
            ),
        )

    def parse_if_statement(self):
        """Parse an if/else statement chain."""
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        then_branch = self.parse_substatement()

        else_branch = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            if self.current_token[0] == "IF":
                else_branch = self.parse_if_statement()
            else:
                else_branch = self.parse_substatement()

        return IfNode(
            condition=condition, then_branch=then_branch, else_branch=else_branch
        )

    def parse_for_statement(self):
        """Parse a C-style ``for`` loop or dispatch to ``for-in`` parsing."""
        self.eat("FOR")
        if self.current_token[0] != "LPAREN":
            return self.parse_for_in_statement_after_for()

        self.enter_value_type_scope()
        try:
            self.eat("LPAREN")

            init = None
            if self.current_token[0] != "SEMICOLON":
                if self.is_variable_declaration():
                    init = self.parse_for_loop_variable_declaration()
                else:
                    # ``k = 0, m = 3`` gives the list of its expressions, like an update.
                    init = self.parse_expression_sequence()
                    if not isinstance(init, list):
                        init = ExpressionStatementNode(init)
                self.eat("SEMICOLON")
            else:
                self.eat("SEMICOLON")

            condition = None
            if self.current_token[0] != "SEMICOLON":
                condition = self.parse_expression()
            self.eat("SEMICOLON")

            update = None
            if self.current_token[0] != "RPAREN":
                update = self.parse_expression_sequence()

            self.eat("RPAREN")
            body = self.parse_substatement()

            return ForNode(init=init, condition=condition, update=update, body=body)
        finally:
            self.restore_value_type_scope()

    def parse_expression_sequence(self):
        """A for-loop update: ``a, b`` and the comma operator ``(a, b)`` both give the list ``[a, b]``.

        The update's value is discarded, so its operands are the same updates run left to right.
        """
        expressions = []
        pending = [self.parse_expression()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            pending.append(self.parse_expression())
        while pending:
            expr = pending.pop(0)
            if isinstance(expr, CommaExpressionNode):
                pending[:0] = list(expr.expressions)
            else:
                expressions.append(expr)
        return expressions[0] if len(expressions) == 1 else expressions

    def parse_for_in_statement_after_for(self):
        """Parse a ``for pattern in iterable`` loop after ``for`` is consumed."""
        pattern = self.current_token[1]
        self.eat("IDENTIFIER")
        binding_type = None
        binding_qualifiers = []
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            binding_qualifiers = self.parse_variable_qualifiers()
            binding_type = self.parse_type()
        self.eat("IN")
        previous_suppression = getattr(self, "suppress_braced_constructor", False)
        self.suppress_braced_constructor = True
        try:
            iterable = self.parse_expression()
        finally:
            self.suppress_braced_constructor = previous_suppression

        self.enter_value_type_scope({pattern: binding_type})
        try:
            body = self.parse_substatement()
        finally:
            self.restore_value_type_scope()

        return ForInNode(
            pattern=pattern,
            iterable=iterable,
            body=body,
            binding_type=binding_type,
            binding_qualifiers=binding_qualifiers,
        )

    def parse_for_loop_variable_declaration(self):
        """Parse variable declarations in for loops (without consuming semicolon)."""
        if (
            self.current_token[0] == "VAR"
            and self.peek()[0] != "LESS_THAN"
            and self.peek(2)[0] in {"COLON", "EQUALS"}
        ):
            self.eat("VAR")
            name = self.parse_binding_identifier()
            var_type = None
            sizes = []
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                var_type = self.parse_type()
                sizes = self.parse_array_suffix_sizes()
            attributes = self.parse_post_declaration_attributes()
            if var_type is not None:
                sizes += self.parse_array_suffix_sizes()
                var_type = self.wrap_array_dimensions(var_type, sizes)

            initial_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                initial_value = self.parse_expression()

            return self.finish_variable_contract(
                VariableNode(
                    name=name,
                    var_type=var_type,
                    initial_value=initial_value,
                    attributes=attributes,
                )
            )

        parsed_qualifiers = self.parse_variable_qualifiers()
        qualifiers, resource_qualifiers = self.partition_resource_qualifiers(
            parsed_qualifiers
        )
        var_type = self.parse_type()
        declarations = []
        while True:
            declarations.append(
                self.parse_variable_declarator(
                    var_type, qualifiers, [], resource_qualifiers
                )
            )
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        # Don't consume semicolon - that's handled by the for loop parser.
        return declarations[0] if len(declarations) == 1 else declarations

    def parse_while_statement(self):
        """Parse a while loop statement."""
        self.eat("WHILE")
        condition = self.parse_expression()
        body = self.parse_substatement()

        return WhileNode(condition=condition, body=body)

    def parse_do_while_statement(self):
        """Parse a do-while loop statement."""
        self.eat("DO")
        body = self.parse_substatement()
        self.eat("WHILE")
        condition = self.parse_expression()
        self.eat("SEMICOLON")

        return DoWhileNode(body=body, condition=condition)

    def parse_loop_statement(self):
        """Parse an unconditional loop statement."""
        self.eat("LOOP")
        body = self.parse_substatement()

        return LoopNode(body=body)

    def parse_match_statement(self):
        """Parse a match statement and all of its arms."""
        self.eat("MATCH")
        previous_suppression = getattr(self, "suppress_braced_constructor", False)
        self.suppress_braced_constructor = True
        try:
            expression = self.parse_expression()
        finally:
            self.suppress_braced_constructor = previous_suppression

        self.eat("LBRACE")
        arms = []

        while self.current_token[0] != "RBRACE":
            arm = self.parse_match_arm()
            arms.append(arm)

        self.eat("RBRACE")

        return MatchNode(expression=expression, arms=arms)

    def parse_match_arm(self):
        """Parse a single match arm."""
        pattern = self.parse_pattern()

        guard = None
        if self.current_token[0] == "IF":
            self.eat("IF")
            guard = self.parse_expression()

        self.eat("FAT_ARROW")
        body = self.parse_match_arm_body()

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")

        return MatchArmNode(pattern=pattern, guard=guard, body=body)

    def parse_match_arm_body(self):
        """Parse a match arm body, including expression arms without semicolons."""
        if self.current_token[0] == "LBRACE":
            return self.parse_block()

        if self.current_token[0] in [
            "IF",
            "FOR",
            "WHILE",
            "DO",
            "LOOP",
            "MATCH",
            "SWITCH",
            "RETURN",
            "BREAK",
            "CONTINUE",
            "LET",
        ]:
            return self.parse_statement()

        expr = self.parse_expression()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ExpressionStatementNode(expr)
        return ExpressionStatementNode(expr, is_tail_expression=True)

    def parse_pattern(self):
        """Parse a match pattern."""
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "_":
            self.eat("IDENTIFIER")
            return WildcardPatternNode()

        if self.current_token[0] == "RANGE":
            self.eat("RANGE")
            return IdentifierPatternNode("..")

        if self.is_pattern_literal_token(self.current_token[0]):
            literal = self.parse_literal()
            return LiteralPatternNode(literal)

        if self.is_pattern_path_token(self.current_token[0]):
            path = self.parse_pattern_path()
            if self.current_token[0] == "LPAREN":
                return self.parse_constructor_pattern(path)
            if self.current_token[0] == "LBRACE":
                return self.parse_struct_pattern(path)
            return IdentifierPatternNode(path)

        literal = self.parse_literal()
        return LiteralPatternNode(literal)

    def is_pattern_literal_token(self, token_type):
        return token_type in [
            "BOOLEAN_LITERAL",
            "NUMBER",
            "FLOAT_NUMBER",
            "HEX_NUMBER",
            "BIN_NUMBER",
            "OCT_NUMBER",
            "STRING_LITERAL",
            "CHAR_LITERAL",
        ]

    def is_pattern_path_token(self, token_type):
        return token_type in [
            "IDENTIFIER",
            "VEC2",
            "VEC3",
            "VEC4",
            "MAT2",
            "MAT3",
            "MAT4",
            "BOOL",
            "INT",
            "UINT",
            "FLOAT",
            "DOUBLE",
        ]

    def parse_pattern_path(self):
        segments = [str(self.current_token[1])]
        self.eat(self.current_token[0])

        while self.current_token[0] == "DOUBLE_COLON":
            self.eat("DOUBLE_COLON")
            segments.append(str(self.current_token[1]))
            self.eat(self.current_token[0])

        return "::".join(segments)

    def parse_constructor_pattern(self, type_name):
        self.eat("LPAREN")
        arguments = []

        while self.current_token[0] != "RPAREN":
            arguments.append(self.parse_pattern())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RPAREN")
        return ConstructorPatternNode(type_name, arguments)

    def parse_struct_pattern(self, type_name):
        self.eat("LBRACE")
        field_patterns = {}
        has_rest = False

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            if self.current_token[0] == "RANGE":
                self.eat("RANGE")
                has_rest = True
                continue

            field_name = str(self.current_token[1])
            self.eat(self.current_token[0])

            if self.current_token[0] == "COLON":
                self.eat("COLON")
                field_patterns[field_name] = self.parse_pattern()
            else:
                field_patterns[field_name] = IdentifierPatternNode(field_name)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RBRACE")
        return StructPatternNode(type_name, field_patterns, has_rest=has_rest)

    def parse_return_statement(self):
        """Parse a return statement with an optional value."""
        self.eat("RETURN")

        value = None
        if self.current_token[0] != "SEMICOLON":
            value = self.parse_expression()
            self.constrain_cooperative_matrix_result(
                value,
                self.current_function_return_type,
                "function return type",
            )

        self.eat("SEMICOLON")
        return ReturnNode(value=value)

    def parse_switch_statement(self):
        """Parse a switch statement and its case clauses."""
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")

        self.eat("LBRACE")
        cases = []

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "CASE":
                case = self.parse_case()
                cases.append(case)
            elif self.current_token[0] == "DEFAULT":
                case = self.parse_default_case()
                cases.append(case)
            else:
                self.reject_unexpected_token(
                    "a switch body before its first case label"
                )

        self.eat("RBRACE")

        if cases and not cases[-1].statements:
            # GLSL ES 3.00 section 6.2 and C++ [stmt.label] (before C++23), which MSL and HLSL follow, need a
            # statement after the last label. Leaving the switch from there is what that label already does.
            cases[-1].statements.append(BreakNode())
        return SwitchNode(expression=expression, cases=cases)

    def parse_case(self):
        """Parse one explicit switch case."""
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            self.append_parsed_nodes(statements, stmt)

        return CaseNode(value=value, statements=statements)

    def parse_default_case(self):
        """Parse the default switch case."""
        self.eat("DEFAULT")
        self.eat("COLON")

        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            self.append_parsed_nodes(statements, stmt)

        return CaseNode(value=None, statements=statements)

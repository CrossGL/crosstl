"""Statements in the Metal parser: blocks, declaration detection, local variables, if, for, range-for, while, do-while, return, switch and case statements."""

from crosstl.translator.refusals import MetalSyntaxError

from ...switch_clauses import record_default_position
from ..local_aggregate_declarators import split_block_aggregate_declaration
from ..MetalAst import (
    AssignmentNode,
    BlockNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    IfNode,
    RangeForNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    TypeAliasNode,
    VariableNode,
    WhileNode,
)
from .parser_tables import (
    IDENTIFIER_TYPE_QUALIFIERS,
    RAYTRACING_TYPE_QUALIFIERS,
    SCOPED_IDENTIFIER_PART_TOKENS,
    SIGNED_TYPE_PREFIXES,
    TYPE_TOKENS,
)


class MetalStatementParsingMixin:
    """Statement members of ``MetalParser``."""

    def parse_block(self):
        statements = []
        initial_scope_names = (
            self.pending_block_scope_names.pop()
            if self.pending_block_scope_names
            else set()
        )
        self.local_variable_scopes.append(set(initial_scope_names))
        self.local_type_scopes.append(set())
        self.eat("LBRACE")
        try:
            while self.current_token[0] != "RBRACE":
                statement = self.parse_statement()
                if isinstance(statement, list):
                    statements.extend(statement)
                elif statement is not None:
                    statements.append(statement)
            self.eat("RBRACE")
            return statements
        finally:
            self.known_types.difference_update(self.local_type_scopes.pop())
            self.local_variable_scopes.pop()

    def parse_statement_body(self):
        if self.current_token[0] == "LBRACE":
            return self.parse_block()
        self.local_type_scopes.append(set())
        try:
            statement = self.parse_statement()
            if isinstance(statement, list):
                return statement
            return [] if statement is None else [statement]
        finally:
            self.known_types.difference_update(self.local_type_scopes.pop())

    def is_declaration_start(self):
        return self.is_declaration_start_at(
            self.skip_leading_attribute_tokens_at(self.pos)
        )

    def is_declaration_start_at(self, idx):
        if idx >= len(self.tokens):
            return False
        token_type, token_value = self.tokens[idx]
        if self.is_scoped_call_expression_start_at(idx):
            return False
        if self.is_templated_call_expression_start_at(idx):
            return False
        if self.is_known_local_variable_name(token_value):
            return False
        if self.is_scoped_type_declaration_start_at(idx):
            return True
        if token_type == "ALIGNAS":
            return True
        if token_type == "STRUCT":
            return (
                idx + 2 < len(self.tokens)
                and self.tokens[idx + 1][0] == "IDENTIFIER"
                and self.tokens[idx + 2][0]
                in {
                    "IDENTIFIER",
                    "MULTIPLY",
                    "BITWISE_AND",
                    "LBRACKET",
                }
            )
        if self.is_qualifier_token_at(idx):
            return True
        if token_type in TYPE_TOKENS:
            if token_type == "IDENTIFIER":
                if self.is_decltype_start_at(idx):
                    return True
                if token_value in SIGNED_TYPE_PREFIXES:
                    return True
                if (
                    token_value
                    in IDENTIFIER_TYPE_QUALIFIERS | RAYTRACING_TYPE_QUALIFIERS
                ):
                    return True
                next_idx = idx + 1
                next_tok = (
                    self.tokens[next_idx][0] if next_idx < len(self.tokens) else "EOF"
                )
                if next_tok in [
                    "IDENTIFIER",
                    "SCOPE",
                    "LESS_THAN",
                    "MULTIPLY",
                    "BITWISE_AND",
                ]:
                    return True
                if next_idx < len(self.tokens) and self.is_name_token_at(next_idx):
                    return True
                while next_idx < len(self.tokens) and self.is_qualifier_token_at(
                    next_idx
                ):
                    next_idx += 1
                if next_idx != idx + 1:
                    next_tok = (
                        self.tokens[next_idx][0]
                        if next_idx < len(self.tokens)
                        else "EOF"
                    )
                    return next_tok in {
                        "IDENTIFIER",
                        "MULTIPLY",
                        "BITWISE_AND",
                        "LBRACKET",
                    }
                return token_value in self.known_types
            return True
        return False

    def is_templated_call_expression_start_at(self, idx):
        if idx + 1 >= len(self.tokens) or self.tokens[idx][0] != "IDENTIFIER":
            return False
        if self.tokens[idx + 1][0] != "LESS_THAN":
            return False
        end = self.skip_template_argument_list_at(idx + 1)
        return end < len(self.tokens) and self.tokens[end][0] in {
            "LPAREN",
            "LBRACE",
        }

    def is_scoped_type_declaration_start_at(self, idx):
        if idx >= len(self.tokens):
            return False
        if self.tokens[idx][0] not in {"SCOPE", "METAL"} and not (
            self.tokens[idx][0] == "IDENTIFIER"
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] == "SCOPE"
        ):
            return False

        idx = self.skip_scoped_type_name_at(idx)
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)
        while idx < len(self.tokens) and self.tokens[idx][0] in {
            "MULTIPLY",
            "BITWISE_AND",
        }:
            idx += 1
        return idx < len(self.tokens) and self.tokens[idx][0] in {
            "IDENTIFIER",
            "LBRACKET",
        }

    def skip_scoped_type_name_at(self, idx):
        if idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
        if idx >= len(self.tokens):
            return idx
        if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
            return idx
        idx += 1
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if idx >= len(self.tokens):
                return idx
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return idx
            idx += 1
        return idx

    def is_known_local_variable_name(self, name):
        return any(name in scope for scope in reversed(self.local_variable_scopes))

    def register_local_variable_name(self, name):
        if self.local_variable_scopes and name:
            self.local_variable_scopes[-1].add(name)

    def register_known_type(self, name):
        if not name:
            return
        if self.local_type_scopes and name not in self.known_types:
            self.local_type_scopes[-1].add(name)
        self.known_types.add(name)

    def is_scoped_call_expression_start_at(self, idx):
        if idx + 2 >= len(self.tokens):
            return False
        if self.tokens[idx][0] not in {"IDENTIFIER", "METAL"}:
            return False
        if self.tokens[idx + 1][0] != "SCOPE":
            return False

        idx += 2
        while idx < len(self.tokens):
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return False
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_balanced_tokens_at(idx, "LESS_THAN", "GREATER_THAN")
            if idx < len(self.tokens) and self.tokens[idx][0] in {
                "LPAREN",
                "LBRACE",
            }:
                return True
            if idx >= len(self.tokens) or self.tokens[idx][0] != "SCOPE":
                break
            idx += 1

        return idx < len(self.tokens) and self.tokens[idx][0] == "LPAREN"

    def parse_statement(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        if self.is_for_loop_macro_prefix():
            self.eat("IDENTIFIER")
            return None
        if self.current_token[0] == "PREPROCESSOR":
            self.parse_preprocessor_directive()
            return None
        if self.current_token == ("IDENTIFIER", "_Pragma"):
            self.parse_pragma_statement()
            return None
        if self.current_token[0] == "USING":
            # Retain body-local ``using X = Y;`` aliases (parse_using_statement
            # returns a TypeAliasNode) so codegen can resolve declarations that
            # reference the alias; ``using namespace``/using-declarations return
            # None and are filtered out by the statement collector.
            return self.parse_using_statement()
        if self.current_token[0] == "TYPEDEF":
            # Retain body-local typedefs for the same reason as using aliases:
            # later declarations and expressions must resolve them in lexical
            # order before CrossGL generation.
            alias = self.parse_typedef()
            return alias if isinstance(alias, (TypeAliasNode, StructNode)) else None
        if self.is_nested_aggregate_declaration_start():
            # A block-scope class definition may also declare objects of it; the
            # importer hoists the class, so its objects become later statements.
            if self.current_token[0] == "STRUCT":
                return split_block_aggregate_declaration(
                    self.parse_struct(allow_unnamed=True)
                )
            if self.current_token[0] == "CLASS":
                return self.parse_class()
            if self.is_union_alias_start():
                return split_block_aggregate_declaration(
                    self.parse_union(allow_unnamed=True)
                )
            self.skip_nested_aggregate_declaration()
            return None
        if self.current_token[0] == "LBRACE":
            return BlockNode(self.parse_block())
        if self.is_statement_expression_block_start():
            return self.parse_statement_expression_block()
        if self.current_token[0] == "ENUM":
            return self.parse_enum()
        if self.is_declaration_start():
            return self.parse_variable_declaration_or_assignment()
        elif self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
        elif self.current_token[0] == "DO":
            return self.parse_do_while_statement()
        elif self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()
        elif self.current_token[0] == "RETURN":
            return self.parse_return_statement()
        elif self.current_token[0] == "BREAK":
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        elif self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return ContinueNode()
        elif self.current_token[0] == "DISCARD":
            self.eat("DISCARD")
            self.eat("SEMICOLON")
            return DiscardNode()
        elif self.current_token[0] == "STATIC_ASSERT":
            return self.parse_static_assert()
        else:
            return self.parse_expression_statement()

    def is_statement_expression_block_start(self):
        return self.current_token[0] == "LPAREN" and self.peek(1)[0] == "LBRACE"

    def parse_statement_expression_block(self):
        self.eat("LPAREN")
        block = BlockNode(self.parse_block())
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return block

    def is_for_loop_macro_prefix(self):
        if self.current_token[0] != "IDENTIFIER" or self.peek(1)[0] != "FOR":
            return False
        name = self.current_token[1]
        return bool(name) and name == name.upper()

    def parse_variable_declaration_or_assignment(self):
        attributes = self.parse_attributes()
        alignas_specs = self.parse_alignas_specifiers()
        attributes.extend(self.parse_attributes())
        vtype, qualifiers = self.parse_type_specifier(attributes=attributes)
        base_vtype = self.base_type_for_remaining_declarators(vtype)
        qualifier_contract = dict(self.last_type_specifier_qualifier_contract)
        name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
        vtype = self.apply_declarator_type_suffix(vtype, type_suffix)
        attributes.extend(self.parse_attributes())

        var_node = VariableNode(
            vtype, name, qualifiers=qualifiers, attributes=attributes
        )
        var_node.pointee_qualifiers = list(qualifier_contract["pointee"])
        var_node.indirection_qualifiers = list(qualifier_contract["indirection"])
        var_node.array_sizes = array_sizes
        self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
        var_node.alignas = alignas_specs
        self.register_local_variable_name(name)
        if "const" in qualifiers or "constexpr" in qualifiers:
            var_node.is_const = True

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return var_node
        if self.current_token[0] == "COMMA":
            return self.parse_remaining_variable_declarations(
                base_vtype, qualifiers, [var_node]
            )

        if self.current_token[0] in [
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
            "ASSIGN_MOD",
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()
            node = AssignmentNode(var_node, value, op)
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_variable_declarations(
                    base_vtype, qualifiers, [node]
                )
            self.eat_statement_semicolon()
            return node

        if self.current_token[0] == "LPAREN":
            node = AssignmentNode(
                var_node,
                self.parse_parenthesized_declaration_initializer(vtype),
            )
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_variable_declarations(
                    base_vtype, qualifiers, [node]
                )
            self.eat("SEMICOLON")
            return node

        if self.current_token[0] == "LBRACE":
            node = AssignmentNode(
                var_node,
                self.parse_braced_declaration_initializer(vtype),
            )
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_variable_declarations(
                    base_vtype, qualifiers, [node]
                )
            self.eat("SEMICOLON")
            return node

        expr = self.parse_expression()
        self.eat_statement_semicolon()
        return expr

    def parse_remaining_variable_declarations(self, vtype, qualifiers, nodes):
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            decl_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            attributes = self.parse_attributes()
            var_node = VariableNode(
                decl_type, name, qualifiers=list(qualifiers), attributes=attributes
            )
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            self.register_local_variable_name(name)
            if "const" in qualifiers or "constexpr" in qualifiers:
                var_node.is_const = True

            if self.current_token[0] in [
                "EQUALS",
                "PLUS_EQUALS",
                "MINUS_EQUALS",
                "MULTIPLY_EQUALS",
                "DIVIDE_EQUALS",
                "ASSIGN_MOD",
                "ASSIGN_AND",
                "ASSIGN_OR",
                "ASSIGN_XOR",
                "ASSIGN_SHIFT_LEFT",
                "ASSIGN_SHIFT_RIGHT",
            ]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                nodes.append(AssignmentNode(var_node, self.parse_expression(), op))
            elif self.current_token[0] == "LPAREN":
                nodes.append(
                    AssignmentNode(
                        var_node,
                        self.parse_parenthesized_declaration_initializer(decl_type),
                    )
                )
            elif self.current_token[0] == "LBRACE":
                nodes.append(
                    AssignmentNode(
                        var_node,
                        self.parse_braced_declaration_initializer(decl_type),
                    )
                )
            else:
                nodes.append(var_node)

        self.eat_statement_semicolon()
        return nodes

    def parse_if_statement(self):
        start_token = self.current_token
        if_chain = []
        else_if_chain = []
        if_constexpr = []
        else_if_constexpr = []
        else_body = None
        self.eat("IF")
        if_constexpr.append(self.parse_optional_if_constexpr())
        self.eat("LPAREN")
        condition = self.parse_expression(allow_comma=True)
        self.eat("RPAREN")
        self.parse_control_statement_attributes()
        body = self.parse_statement_body()
        if_chain.append((condition, body))
        while self.current_token[0] == "ELSE_IF":
            self.eat("ELSE_IF")
            else_if_constexpr.append(self.parse_optional_if_constexpr())
            self.eat("LPAREN")
            condition = self.parse_expression(allow_comma=True)
            self.eat("RPAREN")
            self.parse_control_statement_attributes()
            body = self.parse_statement_body()
            else_if_chain.append((condition, body))

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement_body()

        node = IfNode(
            if_chain=if_chain,
            else_if_chain=else_if_chain,
            else_body=else_body,
            if_constexpr=if_constexpr,
            else_if_constexpr=else_if_constexpr,
        )
        node.source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        return node

    def parse_optional_if_constexpr(self):
        if self.current_token[0] == "CONSTEXPR":
            self.eat("CONSTEXPR")
            return True
        elif self.current_token == ("IDENTIFIER", "IF_CONSTEXPR"):
            self.eat("IDENTIFIER")
            return True
        return False

    def parse_control_statement_attributes(self):
        self.parse_attributes()

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")

        if self.is_range_for_statement():
            return self.parse_range_for_statement()

        init = None
        if self.current_token[0] != "SEMICOLON":
            init = self.parse_for_init()
            if self.current_token[0] == "COMMA":
                init = self.parse_for_tail_items(init, "SEMICOLON")
        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression(allow_comma=True)
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression(allow_comma=True)
            if self.current_token[0] == "COMMA":
                update = self.parse_for_tail_items(update, "RPAREN")
        self.eat("RPAREN")

        self.parse_control_statement_attributes()
        body = self.parse_statement_body()

        return ForNode(init, condition, update, body)

    def parse_for_tail_items(self, first_item, terminator):
        items = [first_item]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if self.current_token[0] == terminator:
                break
            items.append(
                self.parse_for_init()
                if terminator == "SEMICOLON"
                else self.parse_expression()
            )
        return items

    def is_range_for_statement(self):
        idx = self.pos
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "RPAREN" and depth == 0:
                return False
            if token_type == "SEMICOLON" and depth == 0:
                return False
            if token_type == "COLON" and depth == 0:
                return True
            if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "RBRACE"} and depth > 0:
                depth -= 1
            idx += 1
        return False

    def parse_range_for_statement(self):
        vtype, qualifiers = self.parse_type_specifier()
        name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
        self.eat("COLON")
        iterable = self.parse_expression(allow_comma=True)
        self.eat("RPAREN")

        body = self.parse_statement_body()

        node = RangeForNode(
            self.apply_declarator_type_suffix(vtype, type_suffix),
            name,
            iterable,
            body,
            qualifiers=qualifiers,
        )
        node.array_sizes = array_sizes
        self.apply_declarator_metadata(node, type_suffix, grouped_suffix)
        return node

    def parse_for_init(self):
        if self.is_declaration_start():
            vtype, qualifiers = self.parse_type_specifier()
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            init_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            var_node = VariableNode(init_type, name, qualifiers=qualifiers)
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            self.register_local_variable_name(name)
            if "const" in qualifiers or "constexpr" in qualifiers:
                var_node.is_const = True
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                init_value = self.parse_expression()
                return AssignmentNode(var_node, init_value)
            return var_node
        return self.parse_expression(allow_comma=True)

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode(None)
        value = self.parse_expression(allow_comma=True)
        self.eat_statement_semicolon()
        return ReturnNode(value)

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression(allow_comma=True)
        self.eat("RPAREN")
        self.parse_control_statement_attributes()
        body = self.parse_statement_body()
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        body = self.parse_statement_body()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression(allow_comma=True)
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_expression_statement(self):
        expr = self.parse_expression(allow_comma=True)
        self.eat_statement_semicolon()
        return expr

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        self.local_type_scopes.append(set())

        cases = []
        default = None
        default_index = None
        try:
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                if self.current_token[0] == "CASE":
                    cases.append(self.parse_case_statement())
                elif self.current_token[0] == "DEFAULT":
                    self.eat("DEFAULT")
                    self.eat("COLON")
                    default_statements = []
                    default_index = len(cases)

                    # A clause keeps its own jumps: one without a break falls through.
                    while self.current_token[0] not in [
                        "CASE",
                        "DEFAULT",
                        "RBRACE",
                        "EOF",
                    ]:
                        statement = self.parse_statement()
                        if isinstance(statement, list):
                            default_statements.extend(statement)
                        elif statement is not None:
                            default_statements.append(statement)

                    default = default_statements
                else:
                    raise MetalSyntaxError(
                        "Unexpected token in switch statement: "
                        f"{self.current_token[0]}"
                    )

            self.eat("RBRACE")
            return record_default_position(
                SwitchNode(expression, cases, default), default_index
            )
        finally:
            self.known_types.difference_update(self.local_type_scopes.pop())

    def parse_case_statement(self):
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        statements = []

        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            statement = self.parse_statement()
            if isinstance(statement, list):
                statements.extend(statement)
            elif statement is not None:
                statements.append(statement)

        return CaseNode(value, statements)

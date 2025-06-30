from .RustAst import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    WhileNode,
    LoopNode,
    MatchNode,
    MatchArmNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    BreakNode,
    ContinueNode,
    ShaderNode,
    StructNode,
    StructInitializationNode,
    ImplNode,
    TraitNode,
    UnaryOpNode,
    VariableNode,
    LetNode,
    VectorConstructorNode,
    TernaryOpNode,
    UseNode,
    AttributeNode,
    ConstNode,
    StaticNode,
    ArrayAccessNode,
    RangeNode,
    TupleNode,
    ArrayNode,
    ReferenceNode,
    DereferenceNode,
    CastNode,
    BlockNode,
)
from .RustLexer import RustLexer, Lexer, TokenType


class RustParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else None

    def parse(self):
        structs = []
        functions = []
        global_variables = []
        impl_blocks = []
        use_statements = []
        traits = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "USE":
                # Handle use statement
                u = self.parse_use_statement()
                use_statements.append(u)
            elif self.current_token[0] == "STRUCT":
                # Handle struct definition
                s = self.parse_struct()
                structs.append(s)
            elif self.current_token[0] == "IMPL":
                # Handle impl block
                i = self.parse_impl_block()
                impl_blocks.append(i)
            elif self.current_token[0] == "TRAIT":
                # Handle trait definition
                t = self.parse_trait()
                traits.append(t)
            elif self.current_token[0] == "FN":
                # Handle function definition
                f = self.parse_function()
                functions.append(f)
            elif self.current_token[0] == "CONST":
                # Handle const declaration
                c = self.parse_const()
                global_variables.append(c)
            elif self.current_token[0] == "STATIC":
                # Handle static declaration
                s = self.parse_static()
                global_variables.append(s)
            elif self.current_token[0] == "PUB":
                # Handle public items
                visibility = "pub"
                self.eat("PUB")

                if self.current_token[0] == "STRUCT":
                    s = self.parse_struct(visibility=visibility)
                    structs.append(s)
                elif self.current_token[0] == "FN":
                    f = self.parse_function(visibility=visibility)
                    functions.append(f)
                elif self.current_token[0] == "CONST":
                    c = self.parse_const(visibility=visibility)
                    global_variables.append(c)
                elif self.current_token[0] == "STATIC":
                    s = self.parse_static(visibility=visibility)
                    global_variables.append(s)
                elif self.current_token[0] == "TRAIT":
                    t = self.parse_trait(visibility=visibility)
                    traits.append(t)
                elif self.current_token[0] == "USE":
                    u = self.parse_use_statement(visibility=visibility)
                    use_statements.append(u)
                else:
                    self.eat(self.current_token[0])
            elif self.current_token[0] == "POUND":
                # Handle attributes
                attrs = self.parse_attributes()
                # The next item should use these attributes
                if self.current_token[0] == "STRUCT":
                    s = self.parse_struct(attributes=attrs)
                    structs.append(s)
                elif self.current_token[0] == "FN":
                    f = self.parse_function(attributes=attrs)
                    functions.append(f)
                elif self.current_token[0] == "PUB":
                    visibility = "pub"
                    self.eat("PUB")
                    if self.current_token[0] == "STRUCT":
                        s = self.parse_struct(attributes=attrs, visibility=visibility)
                        structs.append(s)
                    elif self.current_token[0] == "FN":
                        f = self.parse_function(attributes=attrs, visibility=visibility)
                        functions.append(f)
                else:
                    self.eat(self.current_token[0])
            else:
                self.eat(self.current_token[0])

        return ShaderNode(
            structs, functions, global_variables, impl_blocks, use_statements
        )

    def eat(self, expected_type):
        if self.current_token[0] == expected_type:
            token = self.current_token
            self.current_index += 1
            if self.current_index < len(self.tokens):
                self.current_token = self.tokens[self.current_index]
            else:
                self.current_token = ("EOF", "")
            return token
        else:
            raise SyntaxError(f"Expected {expected_type}, got {self.current_token[0]}")

    def skip_until(self, token_type):
        while self.current_token[0] != token_type and self.current_token[0] != "EOF":
            self.current_index += 1
            if self.current_index < len(self.tokens):
                self.current_token = self.tokens[self.current_index]
            else:
                self.current_token = ("EOF", "")

    def parse_use_statement(self, visibility=None):
        self.eat("USE")
        path = []

        # Parse the use path
        path.append(self.current_token[1])
        self.eat("IDENTIFIER")

        while self.current_token[0] == "DOUBLE_COLON":
            self.eat("DOUBLE_COLON")
            # Handle glob imports (use module::*)
            if self.current_token[0] == "MULTIPLY":
                path.append("*")
                self.eat("MULTIPLY")
            # Handle braced imports (use module::{item1, item2})
            elif self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                items = []
                while (
                    self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF"
                ):
                    items.append(self.current_token[1])
                    self.eat("IDENTIFIER")
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                    else:
                        break
                self.eat("RBRACE")
                path.append("{" + ", ".join(items) + "}")
            else:
                path.append(self.current_token[1])
                self.eat("IDENTIFIER")
        # Handle alias
        alias = None
        if self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("SEMICOLON")
        return UseNode("::".join(path), alias, visibility)

    def parse_attributes(self):
        attrs = []
        while self.current_token[0] == "POUND":
            self.eat("POUND")
            self.eat("LBRACKET")

            attr_name = self.current_token[1]
            self.eat("IDENTIFIER")

            attr_args = []
            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                while self.current_token[0] != "RPAREN":
                    attr_args.append(self.current_token[1])
                    self.eat(self.current_token[0])
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                self.eat("RPAREN")

            self.eat("RBRACKET")
            attrs.append(AttributeNode(attr_name, attr_args))

        return attrs

    def parse_struct(self, attributes=None, visibility=None):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle generics
        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            # Parse member attributes
            member_attrs = []
            if self.current_token[0] == "POUND":
                member_attrs = self.parse_attributes()

            if self.current_token[0] == "PUB":
                self.eat("PUB")

            # Parse member name
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")

            self.eat("COLON")

            # Parse member type
            member_type = self.parse_type()

            var = VariableNode(member_type, member_name, attributes=member_attrs)
            members.append(var)

            # Handle comma and potential continuation on the same line
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

                # Check if there's another member on the same line
                if self.current_token[0] == "PUB":
                    # Continue parsing the next member without breaking the loop
                    continue
                elif self.current_token[0] == "IDENTIFIER":
                    # Direct member without pub keyword
                    continue
                # If comma is followed by something else, just continue

        self.eat("RBRACE")

        return StructNode(name, members, attributes, visibility, generics)

    def parse_impl_block(self):
        self.eat("IMPL")

        # Handle generics
        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        # Handle trait implementation
        trait_name = None
        struct_name = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "FOR":
            # This is a trait implementation
            trait_name = struct_name
            self.eat("FOR")
            struct_name = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("LBRACE")

        functions = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_token[0] == "FN":
                f = self.parse_function()
                functions.append(f)
            elif self.current_token[0] == "PUB":
                visibility = "pub"
                self.eat("PUB")
                if self.current_token[0] == "FN":
                    f = self.parse_function(visibility=visibility)
                    functions.append(f)
                else:
                    if self.current_token[0] == "EOF":
                        break
                    self.eat(self.current_token[0])
            else:
                if self.current_token[0] == "EOF":
                    break
                self.eat(self.current_token[0])

        self.eat("RBRACE")

        return ImplNode(struct_name, functions, trait_name, generics)

    def parse_trait(self, visibility=None):
        self.eat("TRAIT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle generics
        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        self.eat("LBRACE")

        functions = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_token[0] == "FN":
                f = self.parse_function_signature()  # Traits only have signatures
                functions.append(f)
            else:
                if self.current_token[0] == "EOF":
                    break
                self.eat(self.current_token[0])

        self.eat("RBRACE")

        return TraitNode(name, functions, generics, visibility)

    def parse_generics(self):
        self.eat("LESS_THAN")
        generics = []

        generics.append(self.current_token[1])
        self.eat("IDENTIFIER")

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            generics.append(self.current_token[1])
            self.eat("IDENTIFIER")

        self.eat("GREATER_THAN")
        return generics

    def parse_type(self):
        type_parts = []

        if self.current_token[0] == "AMPERSAND":
            # Reference type
            self.eat("AMPERSAND")
            if self.current_token[0] == "MUT":
                self.eat("MUT")
                type_parts.append("&mut")
            else:
                type_parts.append("&")

        # Basic type
        type_parts.append(self.current_token[1])
        self.eat(self.current_token[0])

        # Handle generic types
        if self.current_token[0] == "LESS_THAN":
            type_parts.append("<")
            self.eat("LESS_THAN")
            type_parts.append(self.parse_type())
            while self.current_token[0] == "COMMA":
                type_parts.append(", ")
                self.eat("COMMA")
                type_parts.append(self.parse_type())
            type_parts.append(">")
            self.eat("GREATER_THAN")

        # Handle array types
        if self.current_token[0] == "LBRACKET":
            type_parts.append("[")
            self.eat("LBRACKET")
            if self.current_token[0] == "NUMBER":
                type_parts.append(self.current_token[1])
                self.eat("NUMBER")
            type_parts.append("]")
            self.eat("RBRACKET")

        return "".join(type_parts)

    def parse_where_clause(self):
        self.eat("WHERE")
        predicates = []

        while self.current_token[0] != "LBRACE" and self.current_token[0] != "EOF":
            # Parse type parameter
            type_param = self.current_token[1]
            self.eat("IDENTIFIER")

            self.eat("COLON")

            # Parse trait bounds
            bounds = []
            bounds.append(self.current_token[1])
            self.eat("IDENTIFIER")

            # Handle multiple bounds with +
            while self.current_token[0] == "PLUS":
                self.eat("PLUS")
                bounds.append(self.current_token[1])
                self.eat("IDENTIFIER")

            predicates.append((type_param, bounds))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            else:
                break

        return predicates

    def parse_struct_initialization(self, struct_name):
        """Parse struct initialization syntax: Name { field: value, ... }"""
        self.eat("LBRACE")
        fields = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            # Parse field name
            field_name = self.current_token[1]
            self.eat("IDENTIFIER")

            self.eat("COLON")

            # Parse field value
            field_value = self.parse_expression()

            fields.append((field_name, field_value))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            else:
                break

        self.eat("RBRACE")
        return StructInitializationNode(struct_name, fields)

    def parse_function(self, attributes=None, visibility=None):
        self.eat("FN")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle generics
        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        params = self.parse_parameters()

        # Parse return type
        return_type = "()"  # Default unit type
        if self.current_token[0] == "ARROW":
            self.eat("ARROW")
            return_type = self.parse_type()

        # Parse optional WHERE clause
        if self.current_token[0] == "WHERE":
            self.parse_where_clause()

        # Parse function body
        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return FunctionNode(
            return_type, name, params, body, attributes, visibility, generics
        )

    def parse_function_signature(self):
        # For trait function signatures (no body)
        self.eat("FN")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle generics
        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        params = self.parse_parameters()

        # Parse return type
        return_type = "()"  # Default unit type
        if self.current_token[0] == "ARROW":
            self.eat("ARROW")
            return_type = self.parse_type()

        self.eat("SEMICOLON")

        return FunctionNode(return_type, name, params, [], [], None, generics)

    def parse_parameters(self):
        self.eat("LPAREN")

        params = []
        if self.current_token[0] != "RPAREN":
            # Handle self parameter
            if self.current_token[0] == "SELF":
                params.append(VariableNode("Self", "self"))
                self.eat("SELF")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
            elif self.current_token[0] == "AMPERSAND":
                # &self or &mut self
                self.eat("AMPERSAND")
                if self.current_token[0] == "MUT":
                    self.eat("MUT")
                    params.append(VariableNode("&mut Self", "self"))
                else:
                    params.append(VariableNode("&Self", "self"))
                self.eat("SELF")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")

            # Parse regular parameters
            while self.current_token[0] != "RPAREN":
                # Parse parameter attributes
                param_attrs = []
                while self.current_token[0] == "POUND":
                    param_attrs.extend(self.parse_attributes())

                # Handle mut keyword
                is_mutable = False
                if self.current_token[0] == "MUT":
                    is_mutable = True
                    self.eat("MUT")

                param_name = self.current_token[1]
                self.eat("IDENTIFIER")
                self.eat("COLON")
                param_type = self.parse_type()

                param = VariableNode(param_type, param_name, is_mutable)
                if param_attrs:
                    param.attributes = param_attrs
                params.append(param)

                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                else:
                    break

        self.eat("RPAREN")
        return params

    def parse_const(self, visibility=None):
        self.eat("CONST")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("COLON")
        const_type = self.parse_type()
        self.eat("EQUALS")
        value = self.parse_expression()
        self.eat("SEMICOLON")

        return ConstNode(name, const_type, value, visibility)

    def parse_static(self, visibility=None):
        self.eat("STATIC")

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("COLON")
        static_type = self.parse_type()
        self.eat("EQUALS")
        value = self.parse_expression()
        self.eat("SEMICOLON")

        return StaticNode(name, static_type, value, is_mutable, visibility)

    def parse_block(self):
        statements = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            try:
                # Let binding
                if self.current_token[0] == "LET":
                    stmt = self.parse_let_statement()
                    statements.append(stmt)
                # Assignment or expression
                elif self.current_token[0] == "IDENTIFIER" or self.current_token[0] in [
                    "VEC2",
                    "VEC3",
                    "VEC4",
                    "MAT2",
                    "MAT3",
                    "MAT4",
                ]:
                    left = self.parse_expression()

                    if self.current_token[0] in [
                        "EQUALS",
                        "PLUS_EQUALS",
                        "MINUS_EQUALS",
                        "MULTIPLY_EQUALS",
                        "DIVIDE_EQUALS",
                        "MOD_EQUALS",
                    ]:
                        op = self.current_token[1]
                        self.eat(self.current_token[0])
                        right = self.parse_expression()
                        self.eat("SEMICOLON")
                        statements.append(AssignmentNode(left, right, op))
                    else:
                        # Expression statement
                        if self.current_token[0] == "SEMICOLON":
                            self.eat("SEMICOLON")
                        statements.append(left)
                # Return statement
                elif self.current_token[0] == "RETURN":
                    self.eat("RETURN")
                    value = None
                    if self.current_token[0] != "SEMICOLON":
                        value = self.parse_expression()
                    self.eat("SEMICOLON")
                    statements.append(ReturnNode(value))
                # Break statement
                elif self.current_token[0] == "BREAK":
                    self.eat("BREAK")
                    label = None
                    value = None
                    if self.current_token[0] != "SEMICOLON":
                        # Could be label or value
                        if self.current_token[0] == "IDENTIFIER":
                            label = self.current_token[1]
                            self.eat("IDENTIFIER")
                    self.eat("SEMICOLON")
                    statements.append(BreakNode(label, value))
                # Continue statement
                elif self.current_token[0] == "CONTINUE":
                    self.eat("CONTINUE")
                    label = None
                    if self.current_token[0] == "IDENTIFIER":
                        label = self.current_token[1]
                        self.eat("IDENTIFIER")
                    self.eat("SEMICOLON")
                    statements.append(ContinueNode(label))
                # If statement
                elif self.current_token[0] == "IF":
                    statements.append(self.parse_if_statement())
                # Match statement
                elif self.current_token[0] == "MATCH":
                    statements.append(self.parse_match_statement())
                # For loop
                elif self.current_token[0] == "FOR":
                    statements.append(self.parse_for_loop())
                # While loop
                elif self.current_token[0] == "WHILE":
                    statements.append(self.parse_while_loop())
                # Loop
                elif self.current_token[0] == "LOOP":
                    statements.append(self.parse_loop())
                else:
                    if self.current_token[0] == "EOF":
                        break
                    self.eat(self.current_token[0])
            except SyntaxError:
                self.skip_until("SEMICOLON")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")

        return statements

    def parse_let_statement(self):
        self.eat("LET")

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Optional type annotation
        var_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            var_type = self.parse_type()

        # Optional initialization
        value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

        self.eat("SEMICOLON")
        return LetNode(name, value, var_type, is_mutable)

    def parse_if_statement(self):
        self.eat("IF")
        condition = self.parse_expression()

        self.eat("LBRACE")
        if_body = self.parse_block()
        self.eat("RBRACE")

        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            if self.current_token[0] == "IF":
                # else if
                else_body = [self.parse_if_statement()]
            else:
                # else block
                self.eat("LBRACE")
                else_body = self.parse_block()
                self.eat("RBRACE")

        return IfNode(condition, if_body, else_body)

    def parse_match_statement(self):
        self.eat("MATCH")
        expression = self.parse_expression()

        self.eat("LBRACE")
        arms = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            # Parse pattern - handle different pattern types
            if self.current_token[0] == "UNDERSCORE":
                pattern = "_"
                self.eat("UNDERSCORE")
            elif self.current_token[0] == "NUMBER":
                pattern = self.current_token[1]
                self.eat("NUMBER")
            elif self.current_token[0] == "STRING":
                pattern = self.current_token[1]
                self.eat("STRING")
            elif self.current_token[0] == "IDENTIFIER":
                pattern = self.current_token[1]
                self.eat("IDENTIFIER")
            else:
                # Fall back to full expression parsing for complex patterns
                pattern = self.parse_expression()

            # Optional guard
            guard = None
            if self.current_token[0] == "IF":
                self.eat("IF")
                guard = self.parse_expression()

            self.eat("FAT_ARROW")

            # Parse arm body
            if self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                body = self.parse_block()
                self.eat("RBRACE")
            else:
                # Single expression
                body = [self.parse_expression()]
                # Handle optional semicolon after expression
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")

            # Handle optional trailing comma
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

            arms.append(MatchArmNode(pattern, guard, body))

        self.eat("RBRACE")
        return MatchNode(expression, arms)

    def parse_for_loop(self):
        self.eat("FOR")
        pattern = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("IN")
        iterable = self.parse_expression()

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return ForNode(pattern, iterable, body)

    def parse_while_loop(self):
        self.eat("WHILE")
        condition = self.parse_expression()

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return WhileNode(condition, body)

    def parse_loop(self):
        self.eat("LOOP")

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return LoopNode(body)

    def parse_expression(self):
        return self.parse_conditional_expression()

    def parse_conditional_expression(self):
        # Handle ternary-like if expressions: if condition { true_expr } else { false_expr }
        if self.current_token[0] == "IF":
            self.eat("IF")
            condition = self.parse_logical_or_expression()
            self.eat("LBRACE")
            true_expr = self.parse_expression()
            self.eat("RBRACE")
            self.eat("ELSE")
            self.eat("LBRACE")
            false_expr = self.parse_expression()
            self.eat("RBRACE")
            return TernaryOpNode(condition, true_expr, false_expr)
        else:
            return self.parse_logical_or_expression()

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        left = self.parse_equality_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        left = self.parse_additive_expression()

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        left = self.parse_range_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_range_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_range_expression(self):
        left = self.parse_multiplicative_expression()

        if self.current_token[0] in ["RANGE", "RANGE_INCLUSIVE"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            return RangeNode(left, right, op == "..=")

        return left

    def parse_multiplicative_expression(self):
        left = self.parse_cast_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MODULO"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_cast_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_cast_expression(self):
        left = self.parse_unary_expression()

        if self.current_token[0] == "AS":
            self.eat("AS")
            target_type = self.parse_type()
            return CastNode(left, target_type)

        return left

    def parse_unary_expression(self):
        if self.current_token[0] in ["MINUS", "EXCLAMATION", "AMPERSAND", "MULTIPLY"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])

            # Handle references specially
            if op == "&":
                is_mutable = False
                if self.current_token[0] == "MUT":
                    is_mutable = True
                    self.eat("MUT")
                expr = self.parse_unary_expression()
                return ReferenceNode(expr, is_mutable)
            elif op == "*":
                expr = self.parse_unary_expression()
                return DereferenceNode(expr)
            else:
                operand = self.parse_unary_expression()
                return UnaryOpNode(op, operand)
        else:
            return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        left = self.parse_primary_expression()

        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                left = MemberAccessNode(left, member)
            elif self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                left = ArrayAccessNode(left, index)
            elif self.current_token[0] == "EXCLAMATION":
                # Macro call (identifier!)
                self.eat("EXCLAMATION")
                self.eat("LPAREN")
                args = []
                while self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                    else:
                        break
                self.eat("RPAREN")
                # Treat macro calls as function calls for code generation
                left = FunctionCallNode(left, args)
            elif self.current_token[0] == "LPAREN":
                # Function call
                self.eat("LPAREN")
                args = []
                while self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                    else:
                        break
                self.eat("RPAREN")
                left = FunctionCallNode(left, args)
            else:
                break

        return left

    def parse_primary_expression(self):
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Check for associated function call (Type::function)
            if self.current_token[0] == "DOUBLE_COLON":
                self.eat("DOUBLE_COLON")
                function_name = self.current_token[1]
                self.eat("IDENTIFIER")

                # Must be followed by function call parentheses
                if self.current_token[0] == "LPAREN":
                    self.eat("LPAREN")
                    args = []
                    while self.current_token[0] != "RPAREN":
                        args.append(self.parse_expression())
                        if self.current_token[0] == "COMMA":
                            self.eat("COMMA")
                        else:
                            break
                    self.eat("RPAREN")
                    # Return as a function call with qualified name
                    return FunctionCallNode(f"{name}::{function_name}", args)
                else:
                    # Path without function call (e.g., Type::CONSTANT)
                    return f"{name}::{function_name}"

            # Check for struct initialization: Name { ... }
            # Only if this identifier is likely a struct constructor (starts with uppercase)
            if self.current_token[0] == "LBRACE" and name[0].isupper():
                return self.parse_struct_initialization(name)

            # Check for vector constructor syntax
            if (
                name in ["Vec2", "Vec3", "Vec4", "Mat2", "Mat3", "Mat4"]
                and self.current_token[0] == "LPAREN"
            ):
                self.eat("LPAREN")
                args = []
                while self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                    else:
                        break
                self.eat("RPAREN")
                return VectorConstructorNode(name, args)

            return name
        elif self.current_token[0] in ["VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4"]:
            # Handle vector types that are tokenized as specific tokens
            name = self.current_token[1]
            self.eat(self.current_token[0])

            # Check for constructor syntax (::new)
            if self.current_token[0] == "DOUBLE_COLON":
                self.eat("DOUBLE_COLON")
                if (
                    self.current_token[0] == "IDENTIFIER"
                    and self.current_token[1] == "new"
                ):
                    self.eat("IDENTIFIER")
                    self.eat("LPAREN")
                    args = []
                    while self.current_token[0] != "RPAREN":
                        args.append(self.parse_expression())
                        if self.current_token[0] == "COMMA":
                            self.eat("COMMA")
                        else:
                            break
                    self.eat("RPAREN")
                    return VectorConstructorNode(name, args)

            return name
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
            self.eat("STRING")
            return value
        elif self.current_token[0] == "TRUE":
            self.eat("TRUE")
            return "true"
        elif self.current_token[0] == "FALSE":
            self.eat("FALSE")
            return "false"
        elif self.current_token[0] == "SELF":
            name = self.current_token[1]
            self.eat("SELF")

            # Check for struct initialization: Self { ... }
            if self.current_token[0] == "LBRACE":
                return self.parse_struct_initialization(name)

            return name
        elif self.current_token[0] == "LPAREN":
            # Parenthesized expression or tuple
            self.eat("LPAREN")
            if self.current_token[0] == "RPAREN":
                # Unit type ()
                self.eat("RPAREN")
                return "()"

            expr = self.parse_expression()

            # Check if it's a tuple
            if self.current_token[0] == "COMMA":
                elements = [expr]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_token[0] != "RPAREN":  # Handle trailing comma
                        elements.append(self.parse_expression())
                self.eat("RPAREN")
                return TupleNode(elements)
            else:
                self.eat("RPAREN")
                return expr
        elif self.current_token[0] == "LBRACKET":
            # Array literal
            self.eat("LBRACKET")
            elements = []
            while self.current_token[0] != "RBRACKET":
                elements.append(self.parse_expression())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                else:
                    break
            self.eat("RBRACKET")
            return ArrayNode(elements)
        elif self.current_token[0] == "LBRACE":
            # Block expression
            self.eat("LBRACE")
            statements = []
            expression = None

            while self.current_token[0] != "RBRACE":
                if self.peek_is_statement():
                    stmt = self.parse_statement()
                    statements.append(stmt)
                else:
                    # Final expression (no semicolon)
                    expression = self.parse_expression()
                    break

            self.eat("RBRACE")
            return BlockNode(statements, expression)
        else:
            raise SyntaxError(
                f"Unexpected token in primary expression: {self.current_token}"
            )

    def peek_is_statement(self):
        # Helper to determine if we're parsing a statement vs expression
        return self.current_token[0] in [
            "LET",
            "IF",
            "MATCH",
            "FOR",
            "WHILE",
            "LOOP",
            "RETURN",
            "BREAK",
            "CONTINUE",
        ]

    def parse_statement(self):
        if self.current_token[0] == "LET":
            return self.parse_let_statement()
        elif self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "MATCH":
            return self.parse_match_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_loop()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_loop()
        elif self.current_token[0] == "LOOP":
            return self.parse_loop()
        elif self.current_token[0] == "RETURN":
            self.eat("RETURN")
            value = None
            if self.current_token[0] != "SEMICOLON":
                value = self.parse_expression()
            self.eat("SEMICOLON")
            return ReturnNode(value)
        elif self.current_token[0] == "BREAK":
            self.eat("BREAK")
            label = None
            value = None
            if self.current_token[0] != "SEMICOLON":
                # Could be label or value
                if self.current_token[0] == "IDENTIFIER":
                    label = self.current_token[1]
                    self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            return BreakNode(label, value)
        elif self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            label = None
            if self.current_token[0] == "IDENTIFIER":
                label = self.current_token[1]
                self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            return ContinueNode(label)
        else:
            # Expression statement
            expr = self.parse_expression()
            self.eat("SEMICOLON")
            return expr

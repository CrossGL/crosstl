"""Parser for Rust source AST construction."""

from .RustAst import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    WhileNode,
    LoopNode,
    MatchNode,
    MatchArmNode,
    MatchBindingPatternNode,
    MatchOrPatternNode,
    MatchRestPatternNode,
    MatchStructPatternNode,
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
    EnumNode,
    EnumVariantNode,
    AssociatedTypeNode,
    TypeAliasNode,
    ImplNode,
    TraitNode,
    UnaryOpNode,
    VariableNode,
    LetNode,
    LetPatternConditionNode,
    MatchesMacroNode,
    ConditionChainNode,
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
    ClosureNode,
    ClosureParameterNode,
    AwaitNode,
    AsyncBlockNode,
    UnsafeBlockNode,
    ConstBlockNode,
    TryNode,
    TryBlockNode,
)
from .RustLexer import RustLexer, Lexer, TokenType


class RustParser:
    """Parse Rust tokens into the Rust backend shader AST."""

    NAME_TOKENS = {
        "IDENTIFIER",
        "VEC2",
        "VEC3",
        "VEC4",
        "MAT2",
        "MAT3",
        "MAT4",
        "OPTION",
        "RESULT",
        "SOME",
        "NONE",
        "OK",
        "ERR",
    }
    PATH_SEGMENT_TOKENS = NAME_TOKENS | {"CRATE", "SELF", "SUPER"}
    ASSIGNMENT_TOKENS = {
        "EQUALS",
        "PLUS_EQUALS",
        "MINUS_EQUALS",
        "MULTIPLY_EQUALS",
        "DIVIDE_EQUALS",
        "MOD_EQUALS",
        "BITWISE_AND_EQUALS",
        "BITWISE_OR_EQUALS",
        "BITWISE_XOR_EQUALS",
        "SHIFT_LEFT_EQUALS",
        "SHIFT_RIGHT_EQUALS",
    }

    def __init__(self, tokens):
        """Initialize the parser with a token stream from ``RustLexer``."""
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else None

    def parse(self):
        """Parse the complete Rust token stream into a shader AST."""
        structs = []
        functions = []
        global_variables = []
        impl_blocks = []
        use_statements = []
        traits = []
        enums = []
        type_aliases = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "USE":
                u = self.parse_use_statement()
                use_statements.append(u)
            elif self.current_token[0] == "TYPE":
                type_aliases.append(self.parse_type_alias())
            elif self.current_token[0] == "STRUCT":
                s = self.parse_struct()
                structs.append(s)
            elif self.current_token[0] == "ENUM":
                e = self.parse_enum()
                enums.append(e)
            elif self.current_token[0] == "IMPL":
                i = self.parse_impl_block()
                impl_blocks.append(i)
            elif self.current_token[0] == "TRAIT":
                t = self.parse_trait()
                traits.append(t)
            elif self.current_starts_function():
                f = self.parse_function_with_qualifiers()
                functions.append(f)
            elif (
                self.current_token[0] == "UNSAFE" and self.peek_token_type() == "EXTERN"
            ):
                self.eat("UNSAFE")
                self.skip_extern_block()
            elif self.current_token[0] == "EXTERN":
                self.skip_extern_block()
            elif self.current_token[0] == "CONST":
                c = self.parse_const()
                global_variables.append(c)
            elif self.current_token[0] == "STATIC":
                s = self.parse_static()
                global_variables.append(s)
            elif self.current_token[0] == "PUB":
                visibility = self.parse_visibility()

                if self.current_token[0] == "STRUCT":
                    s = self.parse_struct(visibility=visibility)
                    structs.append(s)
                elif self.current_starts_function():
                    f = self.parse_function_with_qualifiers(
                        visibility=visibility,
                    )
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
                elif self.current_token[0] == "ENUM":
                    e = self.parse_enum(visibility=visibility)
                    enums.append(e)
                elif self.current_token[0] == "TYPE":
                    type_aliases.append(self.parse_type_alias(visibility=visibility))
                elif self.current_token[0] == "USE":
                    u = self.parse_use_statement(visibility=visibility)
                    use_statements.append(u)
                elif self.current_token[0] == "EXTERN":
                    self.skip_extern_block()
                elif (
                    self.current_token[0] == "UNSAFE"
                    and self.peek_token_type() == "EXTERN"
                ):
                    self.eat("UNSAFE")
                    self.skip_extern_block()
                else:
                    self.eat(self.current_token[0])
            elif self.current_token[0] == "POUND":
                attrs = self.parse_attributes()
                # The next item should use these attributes
                if self.current_token[0] == "STRUCT":
                    s = self.parse_struct(attributes=attrs)
                    structs.append(s)
                elif self.current_token[0] == "ENUM":
                    e = self.parse_enum(attributes=attrs)
                    enums.append(e)
                elif self.current_token[0] == "TYPE":
                    type_aliases.append(self.parse_type_alias(attributes=attrs))
                elif self.current_token[0] == "FN":
                    f = self.parse_function(attributes=attrs)
                    functions.append(f)
                elif self.current_starts_function():
                    f = self.parse_function_with_qualifiers(
                        attributes=attrs,
                    )
                    functions.append(f)
                elif self.current_token[0] == "EXTERN":
                    self.skip_extern_block()
                elif (
                    self.current_token[0] == "UNSAFE"
                    and self.peek_token_type() == "EXTERN"
                ):
                    self.eat("UNSAFE")
                    self.skip_extern_block()
                elif self.current_token[0] == "PUB":
                    visibility = self.parse_visibility()
                    if self.current_token[0] == "STRUCT":
                        s = self.parse_struct(attributes=attrs, visibility=visibility)
                        structs.append(s)
                    elif self.current_token[0] == "ENUM":
                        e = self.parse_enum(attributes=attrs, visibility=visibility)
                        enums.append(e)
                    elif self.current_token[0] == "TYPE":
                        type_aliases.append(
                            self.parse_type_alias(
                                attributes=attrs, visibility=visibility
                            )
                        )
                    elif self.current_starts_function():
                        f = self.parse_function_with_qualifiers(
                            attributes=attrs,
                            visibility=visibility,
                        )
                        functions.append(f)
                    elif self.current_token[0] == "EXTERN":
                        self.skip_extern_block()
                    elif (
                        self.current_token[0] == "UNSAFE"
                        and self.peek_token_type() == "EXTERN"
                    ):
                        self.eat("UNSAFE")
                        self.skip_extern_block()
                else:
                    self.eat(self.current_token[0])
            else:
                self.eat(self.current_token[0])

        return ShaderNode(
            structs,
            functions,
            global_variables,
            impl_blocks,
            use_statements,
            traits,
            enums,
            type_aliases,
        )

    def eat(self, expected_type):
        """Consume and return the current token when it matches ``expected_type``."""
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

    def parse_name_token(self, context="name"):
        if self.current_token[0] not in self.NAME_TOKENS:
            raise SyntaxError(f"Expected {context}, got {self.current_token[0]}")

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def peek_token_type(self, offset=1):
        index = self.current_index + offset
        if index < len(self.tokens):
            return self.tokens[index][0]
        return "EOF"

    def current_starts_function(self):
        index = self.current_index
        saw_qualifier = False

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "FN":
                return True
            if token_type in {"ASYNC", "CONST", "UNSAFE"}:
                saw_qualifier = True
                index += 1
                continue
            if token_type == "EXTERN":
                saw_qualifier = True
                index += 1
                if index < len(self.tokens) and self.tokens[index][0] == "STRING":
                    index += 1
                continue
            break

        return (
            saw_qualifier and index < len(self.tokens) and self.tokens[index][0] == "FN"
        )

    def parse_function_with_qualifiers(
        self,
        attributes=None,
        visibility=None,
    ):
        qualifiers = self.parse_function_qualifiers()
        return self.parse_function(
            attributes=attributes,
            visibility=visibility,
            **qualifiers,
        )

    def parse_function_qualifiers(self):
        qualifiers = {
            "is_async": False,
            "is_const": False,
            "is_unsafe": False,
            "abi": None,
        }

        while self.current_token[0] in {"ASYNC", "CONST", "UNSAFE", "EXTERN"}:
            if self.current_token[0] == "ASYNC":
                qualifiers["is_async"] = True
                self.eat("ASYNC")
            elif self.current_token[0] == "CONST":
                qualifiers["is_const"] = True
                self.eat("CONST")
            elif self.current_token[0] == "UNSAFE":
                qualifiers["is_unsafe"] = True
                self.eat("UNSAFE")
            elif self.current_token[0] == "EXTERN":
                self.eat("EXTERN")
                qualifiers["abi"] = "C"
                if self.current_token[0] == "STRING":
                    qualifiers["abi"] = self.parse_abi_literal()

        return qualifiers

    def parse_abi_literal(self):
        value = self.current_token[1]
        self.eat("STRING")
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            return value[1:-1]
        return value

    def skip_extern_block(self):
        self.eat("EXTERN")
        if self.current_token[0] == "STRING":
            self.eat("STRING")

        if self.current_token[0] == "LBRACE":
            self.skip_balanced_braces()
            return

        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def skip_balanced_braces(self):
        depth = 0
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACE":
                depth += 1
            elif self.current_token[0] == "RBRACE":
                depth -= 1

            token_type = self.current_token[0]
            self.eat(token_type)

            if depth == 0:
                return

    def skip_until(self, token_type):
        while self.current_token[0] != token_type and self.current_token[0] != "EOF":
            self.current_index += 1
            if self.current_index < len(self.tokens):
                self.current_token = self.tokens[self.current_index]
            else:
                self.current_token = ("EOF", "")

    def parse_control_label(self):
        """Parse a Rust loop label or legacy bare identifier label."""
        label = self.current_token[1]
        self.eat(self.current_token[0])
        return label

    def parse_optional_control_label(self):
        if self.current_token[0] in ["LIFETIME", "IDENTIFIER"]:
            return self.parse_control_label()
        return None

    def parse_break_statement(self):
        self.eat("BREAK")
        label = None
        value = None

        if self.current_token[0] != "SEMICOLON":
            if self.current_token[0] == "LIFETIME":
                label = self.parse_control_label()

            if self.current_token[0] != "SEMICOLON":
                value = self.parse_result_expression()

        self.eat("SEMICOLON")
        return BreakNode(label, value)

    def parse_result_expression(self):
        if self.current_token[0] in ["LIFETIME", "LOOP"]:
            return self.parse_loop_expression()
        if self.current_token[0] == "MATCH":
            return self.parse_match_expression()
        return self.parse_expression()

    def parse_return_statement(self):
        self.eat("RETURN")
        value = None
        if self.current_token[0] != "SEMICOLON":
            value = self.parse_result_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_continue_statement(self):
        self.eat("CONTINUE")
        label = self.parse_optional_control_label()
        self.eat("SEMICOLON")
        return ContinueNode(label)

    def parse_labeled_statement(self):
        label = self.parse_control_label()
        self.eat("COLON")

        if self.current_token[0] == "LOOP":
            return self.parse_loop(label)
        if self.current_token[0] == "WHILE":
            return self.parse_while_loop(label)
        if self.current_token[0] == "FOR":
            return self.parse_for_loop(label)

        raise SyntaxError(
            f"Expected loop, while, or for after label, got {self.current_token[0]}"
        )

    def parse_loop_expression(self):
        label = None
        if self.current_token[0] == "LIFETIME":
            label = self.parse_control_label()
            self.eat("COLON")

        if self.current_token[0] != "LOOP":
            raise SyntaxError(f"Expected loop expression, got {self.current_token[0]}")

        return self.parse_loop(label)

    def parse_use_statement(self, visibility=None):
        self.eat("USE")
        path = []
        items = None

        path.append(self.parse_use_path_segment())

        while self.current_token[0] == "DOUBLE_COLON":
            self.eat("DOUBLE_COLON")
            if self.current_token[0] == "MULTIPLY":
                path.append("*")
                self.eat("MULTIPLY")
            elif self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                items = self.parse_use_group_items()
                self.eat("RBRACE")
                path.append(
                    "{"
                    + ", ".join(self.format_use_group_item(item) for item in items)
                    + "}"
                )
            else:
                path.append(self.parse_use_path_segment())
        alias = None
        if self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("SEMICOLON")
        return UseNode("::".join(path), alias, items, visibility)

    def parse_use_group_items(self, prefix=None):
        items = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            items.extend(self.parse_use_group_item(prefix))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        return items

    def parse_use_group_item(self, prefix=None):
        item_path = list(prefix or [])

        if self.current_token[0] == "MULTIPLY":
            item_path.append("*")
            self.eat("MULTIPLY")
            return [{"path": "::".join(item_path), "alias": None}]

        item_path.append(self.parse_use_path_segment())

        while self.current_token[0] == "DOUBLE_COLON":
            self.eat("DOUBLE_COLON")

            if self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                items = self.parse_use_group_items(item_path)
                self.eat("RBRACE")
                return items

            if self.current_token[0] == "MULTIPLY":
                item_path.append("*")
                self.eat("MULTIPLY")
                break

            item_path.append(self.parse_use_path_segment())

        alias = None
        if self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.current_token[1]
            self.eat("IDENTIFIER")

        return [{"path": "::".join(item_path), "alias": alias}]

    def parse_visibility(self):
        self.eat("PUB")
        if self.current_token[0] != "LPAREN":
            return "pub"

        self.eat("LPAREN")
        scope = self.collect_token_text_until({"RPAREN"})
        self.eat("RPAREN")
        return f"pub({scope})"

    def format_use_group_item(self, item):
        if item["alias"]:
            return f"{item['path']} as {item['alias']}"
        return item["path"]

    def parse_use_path_segment(self):
        if self.current_token[0] not in self.PATH_SEGMENT_TOKENS:
            raise SyntaxError(f"Expected use path segment, got {self.current_token[0]}")

        segment = self.current_token[1]
        self.eat(self.current_token[0])
        return segment

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
        name = self.parse_name_token("struct name")

        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        where_clauses = []
        if self.current_token[0] == "WHERE":
            where_clauses = self.parse_where_clause({"LBRACE", "SEMICOLON"})

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return StructNode(name, [], attributes, visibility, generics, where_clauses)

        if self.current_token[0] == "LPAREN":
            members = self.parse_tuple_struct_fields()
            if self.current_token[0] == "WHERE":
                where_clauses = self.parse_where_clause({"SEMICOLON"})
            self.eat("SEMICOLON")
            return StructNode(
                name, members, attributes, visibility, generics, where_clauses
            )

        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            member_attrs = []
            if self.current_token[0] == "POUND":
                member_attrs = self.parse_attributes()

            member_visibility = None
            if self.current_token[0] == "PUB":
                member_visibility = self.parse_visibility()

            member_name = self.parse_name_token("struct member name")

            self.eat("COLON")

            member_type = self.parse_type()

            var = VariableNode(
                member_type,
                member_name,
                attributes=member_attrs,
                visibility=member_visibility,
            )
            members.append(var)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

                if self.current_token[0] == "PUB":
                    continue
                elif self.current_token[0] == "IDENTIFIER":
                    continue

        self.eat("RBRACE")

        return StructNode(
            name, members, attributes, visibility, generics, where_clauses
        )

    def parse_tuple_struct_fields(self):
        self.eat("LPAREN")
        fields = []
        index = 0

        while self.current_token[0] != "RPAREN" and self.current_token[0] != "EOF":
            field_attrs = []
            if self.current_token[0] == "POUND":
                field_attrs = self.parse_attributes()

            field_visibility = None
            if self.current_token[0] == "PUB":
                field_visibility = self.parse_visibility()

            field_type = self.parse_type()
            fields.append(
                VariableNode(
                    field_type,
                    f"field{index}",
                    attributes=field_attrs,
                    visibility=field_visibility,
                )
            )
            index += 1

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RPAREN")
        return fields

    def parse_enum(self, attributes=None, visibility=None):
        self.eat("ENUM")
        name = self.parse_name_token("enum name")

        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        where_clauses = []
        if self.current_token[0] == "WHERE":
            where_clauses = self.parse_where_clause({"LBRACE"})

        self.eat("LBRACE")

        variants = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            variant_attrs = []
            if self.current_token[0] == "POUND":
                variant_attrs = self.parse_attributes()

            variant_name = self.parse_name_token("enum variant name")

            kind = "unit"
            fields = []
            if self.current_token[0] == "LPAREN":
                kind = "tuple"
                fields = self.parse_tuple_variant_fields()
            elif self.current_token[0] == "LBRACE":
                kind = "struct"
                fields = self.parse_struct_variant_fields()

            value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                value = self.parse_expression()

            variants.append(
                EnumVariantNode(variant_name, kind, fields, value, variant_attrs)
            )

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RBRACE")
        return EnumNode(name, variants, attributes, visibility, generics, where_clauses)

    def parse_tuple_variant_fields(self):
        self.eat("LPAREN")
        fields = []
        has_field_metadata = False

        while self.current_token[0] != "RPAREN" and self.current_token[0] != "EOF":
            field_attrs = []
            if self.current_token[0] == "POUND":
                field_attrs = self.parse_attributes()

            field_visibility = None
            if self.current_token[0] == "PUB":
                field_visibility = self.parse_visibility()

            field_type = self.parse_type()
            if field_visibility or field_attrs:
                has_field_metadata = True

            fields.append(
                {
                    "type": field_type,
                    "visibility": field_visibility,
                    "attributes": field_attrs,
                }
            )

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RPAREN")
        if not has_field_metadata:
            return [field["type"] for field in fields]

        return [
            VariableNode(
                field["type"],
                f"field{field_index}",
                attributes=field["attributes"],
                visibility=field["visibility"],
            )
            for field_index, field in enumerate(fields)
        ]

    def parse_struct_variant_fields(self):
        self.eat("LBRACE")
        fields = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            field_attrs = []
            if self.current_token[0] == "POUND":
                field_attrs = self.parse_attributes()

            field_visibility = None
            if self.current_token[0] == "PUB":
                field_visibility = self.parse_visibility()

            field_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("COLON")
            field_type = self.parse_type()
            fields.append(
                VariableNode(
                    field_type,
                    field_name,
                    attributes=field_attrs,
                    visibility=field_visibility,
                )
            )

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RBRACE")
        return fields

    def parse_impl_block(self):
        self.eat("IMPL")

        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        trait_name = None
        struct_name = self.parse_type()

        if self.current_token[0] == "FOR":
            trait_name = struct_name
            self.eat("FOR")
            struct_name = self.parse_type()

        where_clauses = []
        if self.current_token[0] == "WHERE":
            where_clauses = self.parse_where_clause({"LBRACE"})

        self.eat("LBRACE")

        functions = []
        type_aliases = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_starts_function():
                f = self.parse_function_with_qualifiers()
                functions.append(f)
            elif self.current_token[0] == "TYPE":
                type_aliases.append(self.parse_type_alias())
            elif self.current_token[0] == "PUB":
                visibility = self.parse_visibility()
                if self.current_starts_function():
                    f = self.parse_function_with_qualifiers(
                        visibility=visibility,
                    )
                    functions.append(f)
                elif self.current_token[0] == "TYPE":
                    type_aliases.append(self.parse_type_alias(visibility=visibility))
                else:
                    if self.current_token[0] == "EOF":
                        break
                    self.eat(self.current_token[0])
            else:
                if self.current_token[0] == "EOF":
                    break
                self.eat(self.current_token[0])

        self.eat("RBRACE")

        return ImplNode(
            struct_name,
            functions,
            trait_name,
            generics,
            where_clauses,
            type_aliases,
        )

    def parse_trait(self, visibility=None):
        self.eat("TRAIT")
        name = self.parse_name_token("trait name")

        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        where_clauses = []
        if self.current_token[0] == "WHERE":
            where_clauses = self.parse_where_clause({"LBRACE"})

        self.eat("LBRACE")

        functions = []
        associated_types = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_starts_function():
                f = self.parse_trait_function()
                functions.append(f)
            elif self.current_token[0] == "TYPE":
                associated_types.append(self.parse_associated_type())
            else:
                if self.current_token[0] == "EOF":
                    break
                self.eat(self.current_token[0])

        self.eat("RBRACE")

        return TraitNode(
            name,
            functions,
            generics,
            visibility,
            where_clauses,
            associated_types,
        )

    def parse_generics(self):
        self.eat("LESS_THAN")
        generics = []

        while (
            self.current_token[0] != "GREATER_THAN" and self.current_token[0] != "EOF"
        ):
            parameter = self.collect_token_text_until({"COMMA", "GREATER_THAN"})
            if parameter:
                generics.append(parameter)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("GREATER_THAN")
        return generics

    def parse_type(self):
        if self.current_token[0] == "LPAREN":
            return self.parse_tuple_type()

        if self.current_token[0] == "AMPERSAND":
            self.eat("AMPERSAND")
            if self.current_token[0] == "LIFETIME":
                self.eat("LIFETIME")
            if self.current_token[0] == "MUT":
                self.eat("MUT")
                return f"&mut {self.parse_type()}"
            return f"&{self.parse_type()}"

        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            element_type = self.parse_type()
            size = None
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                size = self.parse_array_type_size()
            self.eat("RBRACKET")
            return self.format_array_type(element_type, size)

        type_parts = []

        type_parts.append(self.current_token[1])
        self.eat(self.current_token[0])

        while True:
            if self.current_token[0] == "DOUBLE_COLON":
                type_parts.append("::")
                self.eat("DOUBLE_COLON")
                type_parts.append(self.current_token[1])
                self.eat(self.current_token[0])
            elif self.current_token[0] == "LESS_THAN":
                type_parts.append(self.parse_generic_argument_suffix())
            else:
                break

        if self.current_token[0] == "LBRACKET":
            type_parts.append("[")
            self.eat("LBRACKET")
            if self.current_token[0] == "NUMBER":
                type_parts.append(self.current_token[1])
                self.eat("NUMBER")
            type_parts.append("]")
            self.eat("RBRACKET")

        return "".join(type_parts)

    def parse_tuple_type(self):
        self.eat("LPAREN")

        if self.current_token[0] == "RPAREN":
            self.eat("RPAREN")
            return "()"

        elements = []
        while self.current_token[0] != "RPAREN":
            elements.append(self.parse_type())

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RPAREN")
        return f"({', '.join(elements)})"

    def parse_generic_argument_suffix(self):
        self.eat("LESS_THAN")
        arguments = self.collect_token_text_until({"GREATER_THAN"})
        self.eat("GREATER_THAN")
        return f"<{arguments}>"

    def parse_array_type_size(self):
        parts = []
        depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "RBRACKET" and depth == 0:
                break

            if token_type in ["LPAREN", "LBRACKET", "LBRACE"]:
                depth += 1
            elif token_type in ["RPAREN", "RBRACKET", "RBRACE"]:
                depth -= 1

            parts.append(str(token_value))
            self.eat(token_type)

        return "".join(parts).strip() or None

    def format_array_type(self, element_type, size):
        suffix = f"[{size}]" if size is not None else "[]"
        if "[" not in element_type:
            return f"{element_type}{suffix}"

        base_type, existing_suffix = element_type.split("[", 1)
        return f"{base_type}{suffix}[{existing_suffix}"

    def parse_where_clause(self, terminators=None):
        terminators = set(terminators or {"LBRACE"})
        self.eat("WHERE")
        predicates = []

        while (
            self.current_token[0] not in terminators and self.current_token[0] != "EOF"
        ):
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue

            type_param = self.collect_token_text_until({"COLON", *terminators})
            if not type_param or self.current_token[0] in terminators:
                break

            self.eat("COLON")

            bounds = []
            bound_terminators = {"PLUS", "COMMA", *terminators}
            while (
                self.current_token[0] not in {"COMMA", *terminators}
                and self.current_token[0] != "EOF"
            ):
                bound = self.collect_token_text_until(bound_terminators)
                if bound:
                    bounds.append(bound)
                if self.current_token[0] == "PLUS":
                    self.eat("PLUS")
                    continue
                break

            predicates.append((type_param, bounds))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            else:
                break

        return predicates

    def parse_associated_type(self):
        self.eat("TYPE")
        name = self.parse_name_token("associated type name")

        bounds = []
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            bound_terminators = {"PLUS", "EQUALS", "WHERE", "SEMICOLON"}
            while self.current_token[0] not in {"EQUALS", "WHERE", "SEMICOLON"}:
                bound = self.collect_token_text_until(bound_terminators)
                if bound:
                    bounds.append(bound)
                if self.current_token[0] == "PLUS":
                    self.eat("PLUS")
                    continue
                break

        where_clauses = []
        if self.current_token[0] == "WHERE":
            where_clauses = self.parse_where_clause({"EQUALS", "SEMICOLON"})

        default_type = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            default_type = self.collect_token_text_until({"SEMICOLON"})

        self.eat("SEMICOLON")
        return AssociatedTypeNode(name, bounds, default_type, where_clauses)

    def parse_type_alias(self, visibility=None, attributes=None):
        self.eat("TYPE")
        name = self.parse_name_token("type alias name")

        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        where_clauses = []
        if self.current_token[0] == "WHERE":
            where_clauses = self.parse_where_clause({"EQUALS", "SEMICOLON"})

        alias_type = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            alias_type = self.parse_type()

        self.eat("SEMICOLON")
        return TypeAliasNode(
            name,
            alias_type,
            generics,
            visibility,
            where_clauses,
            attributes,
        )

    def collect_token_text_until(self, terminators):
        parts = []
        depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if depth == 0 and token_type in terminators:
                break

            if token_type in {"LESS_THAN", "LPAREN", "LBRACKET"}:
                depth += 1
            elif token_type == "SHIFT_RIGHT":
                if depth > 1:
                    depth -= 2
                    parts.append(str(token_value))
                    self.eat(token_type)
                    continue
                if depth == 1:
                    depth -= 1
                    parts.append(">")
                    self.split_shift_right_token()
                    continue
            elif token_type in {"GREATER_THAN", "RPAREN", "RBRACKET"}:
                if depth == 0 and token_type in terminators:
                    break
                depth = max(0, depth - 1)

            parts.append(str(token_value))
            self.eat(token_type)

        return self.format_token_parts(parts)

    def split_shift_right_token(self):
        self.tokens[self.current_index] = ("GREATER_THAN", ">")
        self.current_token = self.tokens[self.current_index]

    def format_token_parts(self, parts):
        formatted = []
        previous = None

        for part in parts:
            if part == ",":
                formatted.append(", ")
            else:
                if self.needs_token_part_space(previous, part):
                    formatted.append(" ")
                formatted.append(part)
            previous = part

        return "".join(formatted).strip()

    def needs_token_part_space(self, previous, current):
        if previous is None:
            return False
        if previous in {"<", "[", "(", "::", ","}:
            return False
        if current in {">", ">>", "]", ")", ",", "::", "<", "[", "(", ":"}:
            return False
        if previous in {":", "+", "=", "->", "=>"}:
            return True
        if current in {"+", "=", "->", "=>"}:
            return True
        return self.is_token_word(previous) and self.is_token_word(current)

    def is_token_word(self, part):
        return part.replace("_", "").isalnum()

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

    def parse_function(
        self,
        attributes=None,
        visibility=None,
        is_async=False,
        is_unsafe=False,
        abi=None,
        is_const=False,
    ):
        name, generics, params, return_type, where_clauses = self.parse_function_header(
            {"LBRACE"}
        )
        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return FunctionNode(
            return_type,
            name,
            params,
            body,
            attributes,
            visibility,
            generics,
            where_clauses,
            is_async,
            is_unsafe,
            abi,
            is_const,
        )

    def parse_function_header(self, where_terminators):
        self.eat("FN")
        name = self.parse_name_token("function name")

        generics = []
        if self.current_token[0] == "LESS_THAN":
            generics = self.parse_generics()

        params = self.parse_parameters()

        return_type = "()"
        if self.current_token[0] == "ARROW":
            self.eat("ARROW")
            return_type = self.parse_type()

        where_clauses = []
        if self.current_token[0] == "WHERE":
            where_clauses = self.parse_where_clause(where_terminators)

        return name, generics, params, return_type, where_clauses

    def parse_trait_function(self):
        qualifiers = self.parse_function_qualifiers()
        name, generics, params, return_type, where_clauses = self.parse_function_header(
            {"LBRACE", "SEMICOLON"}
        )

        body = []
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            body = self.parse_block()
            self.eat("RBRACE")
        else:
            self.eat("SEMICOLON")

        return FunctionNode(
            return_type,
            name,
            params,
            body,
            [],
            None,
            generics,
            where_clauses,
            qualifiers["is_async"],
            qualifiers["is_unsafe"],
            qualifiers["abi"],
            qualifiers["is_const"],
        )

    def parse_parameters(self):
        self.eat("LPAREN")

        params = []
        if self.current_token[0] != "RPAREN":
            if self.current_token[0] == "SELF":
                params.append(VariableNode("Self", "self"))
                self.eat("SELF")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
            elif self.current_token[0] == "AMPERSAND":
                self.eat("AMPERSAND")
                if self.current_token[0] == "MUT":
                    self.eat("MUT")
                    params.append(VariableNode("&mut Self", "self"))
                else:
                    params.append(VariableNode("&Self", "self"))
                self.eat("SELF")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")

            while self.current_token[0] != "RPAREN":
                param_attrs = []
                while self.current_token[0] == "POUND":
                    param_attrs.extend(self.parse_attributes())

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
            if self.current_token[0] == "LIFETIME":
                statements.append(self.parse_labeled_statement())
            elif self.current_token[0] == "LET":
                stmt = self.parse_let_statement()
                statements.append(stmt)
            elif (
                self.current_token[0] == "CONST" and self.peek_token_type() != "LBRACE"
            ):
                statements.append(self.parse_const())
            elif self.current_token[0] == "STATIC":
                statements.append(self.parse_static())
            elif (
                self.current_token[0] == "IDENTIFIER"
                or self.current_token[0]
                in [
                    "VEC2",
                    "VEC3",
                    "VEC4",
                    "MAT2",
                    "MAT3",
                    "MAT4",
                    "ASYNC",
                    "UNSAFE",
                ]
                or (
                    self.current_token[0] == "CONST"
                    and self.peek_token_type() == "LBRACE"
                )
            ):
                expression = self.parse_expression()
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                statements.append(expression)
            elif self.current_token[0] == "RETURN":
                statements.append(self.parse_return_statement())
            elif self.current_token[0] == "BREAK":
                statements.append(self.parse_break_statement())
            elif self.current_token[0] == "CONTINUE":
                statements.append(self.parse_continue_statement())
            elif self.current_token[0] == "IF":
                statements.append(self.parse_if_statement())
            elif self.current_token[0] == "MATCH":
                statements.append(self.parse_match_statement())
            elif self.current_token[0] == "FOR":
                statements.append(self.parse_for_loop())
            elif self.current_token[0] == "WHILE":
                statements.append(self.parse_while_loop())
            elif self.current_token[0] == "LOOP":
                statements.append(self.parse_loop())
            else:
                if self.current_token[0] == "EOF":
                    break
                self.eat(self.current_token[0])

        return statements

    def parse_let_statement(self):
        self.eat("LET")

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        name = self.parse_match_pattern()

        var_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            var_type = self.parse_type()

        value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_result_expression()

        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            self.eat("LBRACE")
            else_body = self.parse_block()
            self.eat("RBRACE")

        self.eat("SEMICOLON")
        return LetNode(name, value, var_type, is_mutable, else_body)

    def parse_let_pattern(self):
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return name

        if self.current_token[0] == "UNDERSCORE":
            self.eat("UNDERSCORE")
            return "_"

        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            elements = []

            while self.current_token[0] != "RPAREN":
                elements.append(self.parse_let_pattern())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break

            self.eat("RPAREN")
            return TupleNode(elements)

        raise SyntaxError(f"Expected let pattern, got {self.current_token[0]}")

    def parse_if_statement(self):
        self.eat("IF")
        condition = self.parse_if_condition()

        self.eat("LBRACE")
        if_body = self.parse_block()
        self.eat("RBRACE")

        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            if self.current_token[0] == "IF":
                else_body = [self.parse_if_statement()]
            else:
                # else block
                self.eat("LBRACE")
                else_body = self.parse_block()
                self.eat("RBRACE")

        return IfNode(condition, if_body, else_body)

    def parse_if_condition(self):
        if self.current_token[0] == "LET" or self.has_top_level_logical_and_let():
            return self.parse_condition_chain()
        return self.parse_expression()

    def has_top_level_logical_and_let(self):
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        index = self.current_index

        while index < len(self.tokens):
            token_type = self.tokens[index][0]

            if (
                paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and token_type == "LBRACE"
            ):
                return False

            if (
                paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and token_type == "LOGICAL_AND"
                and index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "LET"
            ):
                return True

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN":
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET":
                bracket_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE":
                if brace_depth == 0:
                    return False
                brace_depth -= 1
            elif token_type == "EOF":
                return False

            index += 1

        return False

    def parse_condition_chain(self):
        operands = [self.parse_condition_chain_operand()]

        while self.current_token[0] == "LOGICAL_AND":
            self.eat("LOGICAL_AND")
            operands.append(self.parse_condition_chain_operand())

        if len(operands) == 1:
            return operands[0]
        return ConditionChainNode(operands)

    def parse_condition_chain_operand(self):
        if self.current_token[0] == "LET":
            return self.parse_let_pattern_condition(stop_at_logical_and=True)
        return self.parse_condition_chain_expression_operand()

    def parse_condition_chain_expression_operand(self):
        left = self.parse_bitwise_or_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_let_pattern_condition(self, stop_at_logical_and=False):
        self.eat("LET")
        pattern = self.parse_match_pattern()
        self.eat("EQUALS")
        if stop_at_logical_and:
            expression = self.parse_condition_chain_expression_operand()
        else:
            expression = self.parse_expression()
        return LetPatternConditionNode(pattern, expression)

    def parse_match_statement(self):
        return self.parse_match(result_mode=False)

    def parse_match_expression(self):
        return self.parse_match(result_mode=True)

    def parse_match(self, result_mode=False):
        self.eat("MATCH")
        expression = self.parse_expression()

        self.eat("LBRACE")
        arms = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            pattern = self.parse_match_pattern()

            guard = None
            if self.current_token[0] == "IF":
                self.eat("IF")
                guard = self.parse_expression()

            self.eat("FAT_ARROW")

            if result_mode and self.current_token[0] == "LBRACE":
                body = self.parse_block_expression()
            elif self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                body = self.parse_block()
                self.eat("RBRACE")
            elif result_mode:
                body = self.parse_result_expression()
            else:
                body = [self.parse_expression()]
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

            arms.append(MatchArmNode(pattern, guard, body))

        self.eat("RBRACE")
        return MatchNode(expression, arms)

    def parse_match_pattern(self):
        patterns = [self.parse_single_match_pattern()]
        while self.current_token[0] == "PIPE":
            self.eat("PIPE")
            patterns.append(self.parse_single_match_pattern())

        if len(patterns) == 1:
            return patterns[0]
        return MatchOrPatternNode(patterns)

    def parse_single_match_pattern(self):
        if self.current_token[0] in {"REF", "MUT"}:
            return self.parse_match_binding_modifier_pattern()

        if (
            self.current_token[0] in {"IDENTIFIER", "UNDERSCORE"}
            and self.peek_token_type() == "AT"
        ):
            name = self.current_token[1]
            self.eat(self.current_token[0])
            self.eat("AT")
            return MatchBindingPatternNode(name, self.parse_single_match_pattern())

        pattern = self.parse_match_pattern_atom()

        if self.current_token[0] in ["RANGE", "RANGE_INCLUSIVE"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            end = self.parse_match_pattern_atom()
            return RangeNode(pattern, end, op == "..=")

        return pattern

    def parse_match_binding_modifier_pattern(self):
        if self.current_token[0] == "REF":
            self.eat("REF")
            if self.current_token[0] == "MUT":
                self.eat("MUT")
        elif self.current_token[0] == "MUT":
            self.eat("MUT")

        if self.current_token[0] not in {"IDENTIFIER", "UNDERSCORE"}:
            raise SyntaxError(
                f"Expected binding after pattern modifier, got {self.current_token[0]}"
            )

        name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "AT":
            self.eat("AT")
            return MatchBindingPatternNode(name, self.parse_single_match_pattern())

        return name

    def parse_match_pattern_atom(self):
        literal_tokens = {
            "NUMBER",
            "STRING",
            "BYTE_STRING",
            "BYTE_RAW_STRING",
            "RAW_STRING",
            "BYTE_CHAR",
            "CHAR_LIT",
        }

        if self.current_token[0] == "UNDERSCORE":
            pattern = "_"
            self.eat("UNDERSCORE")
            return pattern

        if self.current_token[0] == "RANGE":
            self.eat("RANGE")
            return MatchRestPatternNode()

        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            if self.current_token[0] == "RPAREN":
                self.eat("RPAREN")
                return "()"

            pattern = self.parse_match_pattern()
            if self.current_token[0] == "COMMA":
                elements = [pattern]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_token[0] == "RPAREN":
                        break
                    elements.append(self.parse_match_pattern())
                self.eat("RPAREN")
                return TupleNode(elements)

            self.eat("RPAREN")
            return pattern

        if self.current_token[0] == "LBRACKET":
            return self.parse_match_array_pattern()

        if self.current_token[0] == "AMPERSAND":
            self.eat("AMPERSAND")
            is_mutable = False
            if self.current_token[0] == "MUT":
                self.eat("MUT")
                is_mutable = True
            return ReferenceNode(self.parse_single_match_pattern(), is_mutable)

        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            return UnaryOpNode("-", self.parse_match_pattern_atom())

        if self.current_token[0] in literal_tokens:
            pattern = self.current_token[1]
            self.eat(self.current_token[0])
            return pattern

        if self.current_token[0] == "TRUE":
            self.eat("TRUE")
            return "true"

        if self.current_token[0] == "FALSE":
            self.eat("FALSE")
            return "false"

        if self.current_token[0] in self.match_pattern_path_tokens():
            if self.peek_token_type() in {"LPAREN", "DOUBLE_COLON", "LBRACE"}:
                return self.parse_match_path_or_call()

            pattern = self.current_token[1]
            self.eat(self.current_token[0])
            return pattern

        return self.parse_expression()

    def parse_match_array_pattern(self):
        self.eat("LBRACKET")
        elements = []

        while self.current_token[0] != "RBRACKET":
            elements.append(self.parse_match_pattern())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RBRACKET")
        return ArrayNode(elements)

    def parse_match_struct_pattern(self, name):
        self.eat("LBRACE")
        fields = []
        has_rest = False

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "RANGE":
                self.eat("RANGE")
                has_rest = True
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break

            if self.current_token[0] in {"REF", "MUT"}:
                field_name = self.parse_match_field_binding_name()
            else:
                field_name = self.current_token[1]
                self.eat("IDENTIFIER")

            if self.current_token[0] == "COLON":
                self.eat("COLON")
                field_pattern = self.parse_match_pattern()
            else:
                field_pattern = field_name

            fields.append((field_name, field_pattern))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RBRACE")
        return MatchStructPatternNode(name, fields, has_rest)

    def parse_match_field_binding_name(self):
        if self.current_token[0] == "REF":
            self.eat("REF")
            if self.current_token[0] == "MUT":
                self.eat("MUT")
        elif self.current_token[0] == "MUT":
            self.eat("MUT")

        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected field binding after pattern modifier, got {self.current_token[0]}"
            )

        field_name = self.current_token[1]
        self.eat("IDENTIFIER")
        return field_name

    def match_pattern_path_tokens(self):
        return {
            "IDENTIFIER",
            "CRATE",
            "SELF",
            "SUPER",
            "VEC2",
            "VEC3",
            "VEC4",
            "MAT2",
            "MAT3",
            "MAT4",
            "OPTION",
            "RESULT",
            "SOME",
            "NONE",
            "OK",
            "ERR",
        }

    def parse_match_path_or_call(self):
        first_segment = self.current_token[1]
        self.eat(self.current_token[0])
        path = self.parse_path_expression(first_segment)

        if self.current_token[0] == "LPAREN":
            return FunctionCallNode(path, self.parse_match_pattern_arguments())
        if self.current_token[0] == "LBRACE":
            return self.parse_match_struct_pattern(path)

        return path

    def parse_match_pattern_arguments(self):
        self.eat("LPAREN")
        args = []

        while self.current_token[0] != "RPAREN":
            args.append(self.parse_match_pattern())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RPAREN")
        return args

    def parse_for_loop(self, label=None):
        self.eat("FOR")
        pattern = self.parse_let_pattern()
        self.eat("IN")
        iterable = self.parse_expression()

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return ForNode(pattern, iterable, body, label)

    def parse_while_loop(self, label=None):
        self.eat("WHILE")
        condition = self.parse_if_condition()

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return WhileNode(condition, body, label)

    def parse_loop(self, label=None):
        self.eat("LOOP")

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return LoopNode(body, label)

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        left = self.parse_conditional_expression()

        if self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op = self.current_token[1]
            op_token = self.current_token[0]
            self.eat(op_token)
            if op_token == "EQUALS":
                right = self.parse_result_expression()
            else:
                right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def parse_conditional_expression(self):
        if self.current_token[0] == "IF":
            return self.parse_if_expression()
        return self.parse_range_expression()

    def parse_if_expression(self):
        self.eat("IF")
        condition = self.parse_if_condition()
        true_block = self.parse_block_expression()
        self.eat("ELSE")

        if self.current_token[0] == "IF":
            false_block = BlockNode([], self.parse_if_expression())
        else:
            false_block = self.parse_block_expression()

        true_expr = self.get_simple_block_expression(true_block)
        false_expr = self.get_simple_block_expression(false_block)
        if (
            not isinstance(condition, LetPatternConditionNode)
            and not isinstance(condition, ConditionChainNode)
            and true_expr is not None
            and false_expr is not None
        ):
            return TernaryOpNode(condition, true_expr, false_expr)

        return IfNode(condition, true_block, false_block)

    def get_simple_block_expression(self, block):
        expression = block.expression
        if block.statements or isinstance(
            expression, (LoopNode, IfNode, MatchNode, BlockNode)
        ):
            return None
        return expression

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()

        while self.current_token[0] == "PIPE":
            op = self.current_token[1]
            self.eat("PIPE")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()

        while self.current_token[0] == "CARET":
            op = self.current_token[1]
            self.eat("CARET")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()

        while self.current_token[0] == "AMPERSAND":
            op = self.current_token[1]
            self.eat("AMPERSAND")
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
        left = self.parse_shift_expression()

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_shift_expression(self):
        left = self.parse_additive_expression()

        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        left = self.parse_multiplicative_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_range_expression(self):
        if self.current_token[0] in ["RANGE", "RANGE_INCLUSIVE"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            if op == "..=" and self.is_range_expression_boundary():
                raise SyntaxError("Expected range end after ..=")
            end = (
                None
                if self.is_range_expression_boundary()
                else self.parse_logical_or_expression()
            )
            return RangeNode(None, end, op == "..=")

        left = self.parse_logical_or_expression()

        if self.current_token[0] in ["RANGE", "RANGE_INCLUSIVE"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            if op == "..=" and self.is_range_expression_boundary():
                raise SyntaxError("Expected range end after ..=")
            right = (
                None
                if self.is_range_expression_boundary()
                else self.parse_logical_or_expression()
            )
            return RangeNode(left, right, op == "..=")

        return left

    def is_range_expression_boundary(self):
        return self.current_token[0] in {
            "COMMA",
            "FAT_ARROW",
            "LBRACE",
            "RBRACE",
            "RBRACKET",
            "RPAREN",
            "SEMICOLON",
        }

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

        while self.current_token[0] == "AS":
            self.eat("AS")
            target_type = self.parse_type()
            left = CastNode(target_type, left)

        return left

    def parse_unary_expression(self):
        if self.current_token[0] in {"PIPE", "LOGICAL_OR", "MOVE"}:
            return self.parse_closure_expression()

        if self.current_token[0] in ["MINUS", "EXCLAMATION", "AMPERSAND", "MULTIPLY"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])

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

    def parse_closure_expression(self):
        is_move = False
        if self.current_token[0] == "MOVE":
            is_move = True
            self.eat("MOVE")

        if self.current_token[0] == "LOGICAL_OR":
            self.eat("LOGICAL_OR")
            params = []
        else:
            self.eat("PIPE")
            end_index = self.find_closure_parameter_end()
            params = []
            while self.current_index < end_index:
                params.append(self.parse_closure_parameter())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_index >= end_index:
                        break
                    continue
                if self.current_index < end_index:
                    raise SyntaxError(
                        f"Expected comma or closure delimiter, got {self.current_token[0]}"
                    )
                break
            self.eat("PIPE")

        return_type = None
        if self.current_token[0] == "ARROW":
            self.eat("ARROW")
            return_type = self.parse_type()

        body = self.parse_result_expression()
        return ClosureNode(params, body, is_move, return_type)

    def find_closure_parameter_end(self):
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        for index in range(self.current_index, len(self.tokens)):
            token_type = self.tokens[index][0]
            if (
                token_type == "PIPE"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                return index

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN":
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET":
                bracket_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE":
                brace_depth -= 1

        raise SyntaxError("Unterminated closure parameter list")

    def parse_closure_parameter(self):
        pattern = self.parse_single_match_pattern()
        param_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            param_type = self.parse_type()
        return ClosureParameterNode(pattern, param_type)

    def parse_postfix_expression(self):
        left = self.parse_primary_expression()

        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                if self.current_token[0] == "AWAIT":
                    self.eat("AWAIT")
                    left = AwaitNode(left)
                    continue

                if self.current_token[0] not in {"IDENTIFIER", "NUMBER"}:
                    raise SyntaxError(
                        f"Expected IDENTIFIER or NUMBER, got {self.current_token[0]}"
                    )
                member = self.current_token[1]
                self.eat(self.current_token[0])
                left = MemberAccessNode(left, member)
            elif self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                left = ArrayAccessNode(left, index)
            elif self.current_token[0] == "EXCLAMATION":
                self.eat("EXCLAMATION")
                if left == "matches":
                    left = self.parse_matches_macro_expression()
                else:
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
            elif self.current_token[0] == "QUESTION":
                self.eat("QUESTION")
                left = TryNode(left)
            else:
                break

        return left

    def parse_matches_macro_expression(self):
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("COMMA")
        pattern = self.parse_match_pattern()

        guard = None
        if self.current_token[0] == "IF":
            self.eat("IF")
            guard = self.parse_expression()

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")

        self.eat("RPAREN")
        return MatchesMacroNode(expression, pattern, guard)

    def parse_primary_expression(self):
        if self.current_token[0] == "TRY":
            return self.parse_try_block_expression()

        if self.current_token[0] == "ASYNC":
            return self.parse_async_block_expression()

        if self.current_token[0] == "UNSAFE":
            return self.parse_unsafe_block_expression()

        if self.current_token[0] == "CONST" and self.peek_token_type() == "LBRACE":
            return self.parse_const_block_expression()

        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "DOUBLE_COLON":
                return self.finish_path_or_call(name)

            # Only if this identifier is likely a struct constructor (starts with uppercase)
            if self.current_token[0] == "LBRACE" and name[0].isupper():
                return self.parse_struct_initialization(name)

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
        elif self.current_token[0] in [
            "VEC2",
            "VEC3",
            "VEC4",
            "MAT2",
            "MAT3",
            "MAT4",
            "OPTION",
            "RESULT",
            "SOME",
            "NONE",
            "OK",
            "ERR",
        ]:
            name = self.current_token[1]
            self.eat(self.current_token[0])

            if self.current_token[0] == "DOUBLE_COLON":
                return self.finish_path_or_call(name)

            return name
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
            self.eat("STRING")
            return value
        elif self.current_token[0] == "BYTE_STRING":
            value = self.current_token[1]
            self.eat("BYTE_STRING")
            return value
        elif self.current_token[0] == "BYTE_RAW_STRING":
            value = self.current_token[1]
            self.eat("BYTE_RAW_STRING")
            return value
        elif self.current_token[0] == "RAW_STRING":
            value = self.current_token[1]
            self.eat("RAW_STRING")
            return value
        elif self.current_token[0] == "BYTE_CHAR":
            value = self.current_token[1]
            self.eat("BYTE_CHAR")
            return value
        elif self.current_token[0] == "CHAR_LIT":
            value = self.current_token[1]
            self.eat("CHAR_LIT")
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

            if self.current_token[0] == "DOUBLE_COLON":
                return self.finish_path_or_call(name)

            return name
        elif self.current_token[0] in {"CRATE", "SUPER"}:
            name = self.current_token[1]
            self.eat(self.current_token[0])

            if self.current_token[0] == "DOUBLE_COLON":
                return self.finish_path_or_call(name)

            return name
        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            if self.current_token[0] == "RPAREN":
                self.eat("RPAREN")
                return "()"

            expr = self.parse_result_expression()

            if self.current_token[0] == "COMMA":
                elements = [expr]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_token[0] != "RPAREN":  # Handle trailing comma
                        elements.append(self.parse_result_expression())
                self.eat("RPAREN")
                return TupleNode(elements)
            else:
                self.eat("RPAREN")
                return expr
        elif self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] == "RBRACKET":
                self.eat("RBRACKET")
                return ArrayNode([])

            first_element = self.parse_expression()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                size = self.parse_expression()
                self.eat("RBRACKET")
                return ArrayNode([first_element], size)

            elements = [first_element]
            while self.current_token[0] != "RBRACKET":
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_token[0] != "RBRACKET":
                        elements.append(self.parse_expression())
                else:
                    break
            self.eat("RBRACKET")
            return ArrayNode(elements)
        elif self.current_token[0] == "LBRACE":
            return self.parse_block_expression()
        else:
            raise SyntaxError(
                f"Unexpected token in primary expression: {self.current_token}"
            )

    def parse_try_block_expression(self):
        self.eat("TRY")
        return TryBlockNode(self.parse_block_expression())

    def parse_async_block_expression(self):
        self.eat("ASYNC")
        is_move = False
        if self.current_token[0] == "MOVE":
            is_move = True
            self.eat("MOVE")
        return AsyncBlockNode(self.parse_block_expression(), is_move)

    def parse_unsafe_block_expression(self):
        self.eat("UNSAFE")
        return UnsafeBlockNode(self.parse_block_expression())

    def parse_const_block_expression(self):
        self.eat("CONST")
        return ConstBlockNode(self.parse_block_expression())

    def parse_block_expression(self):
        self.eat("LBRACE")
        statements = []
        expression = None

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue

            if self.current_token[0] in ["IF", "MATCH", "LOOP", "LIFETIME"]:
                final_expression = self.try_parse_block_final_expression()
                if final_expression is not None:
                    expression = final_expression
                    break

            if self.peek_is_statement():
                stmt = self.parse_statement()
                if self.current_token[0] == "RBRACE" and isinstance(stmt, LoopNode):
                    expression = stmt
                else:
                    if self.current_token[0] == "SEMICOLON":
                        self.eat("SEMICOLON")
                    statements.append(stmt)
            else:
                parsed_expression = self.parse_expression()
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                    statements.append(parsed_expression)
                    continue

                expression = parsed_expression
                break

        self.eat("RBRACE")
        return BlockNode(statements, expression)

    def try_parse_block_final_expression(self):
        start_index = self.current_index
        start_token = self.current_token

        try:
            expression = self.parse_result_expression()
            if self.current_token[0] == "RBRACE":
                return expression
        except SyntaxError:
            pass

        self.current_index = start_index
        self.current_token = start_token
        return None

    def parse_path_expression(self, first_segment):
        segments = [first_segment]

        while self.current_token[0] == "DOUBLE_COLON":
            self.eat("DOUBLE_COLON")

            if self.current_token[0] == "LESS_THAN":
                segments[-1] += self.parse_generic_argument_suffix()
                continue

            segments.append(self.current_token[1])
            self.eat(self.current_token[0])

        return "::".join(segments)

    def finish_path_or_call(self, first_segment):
        path = self.parse_path_expression(first_segment)
        if self.current_token[0] == "LPAREN":
            return FunctionCallNode(path, self.parse_call_arguments())
        return path

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []

        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            else:
                break

        self.eat("RPAREN")
        return args

    def peek_is_statement(self):
        if self.current_token[0] == "CONST":
            return self.peek_token_type() != "LBRACE"
        if self.current_token[0] == "STATIC":
            return True

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
            "LIFETIME",
        ]

    def parse_statement(self):
        if self.current_token[0] == "LIFETIME":
            return self.parse_labeled_statement()
        elif self.current_token[0] == "LET":
            return self.parse_let_statement()
        elif self.current_token[0] == "CONST":
            return self.parse_const()
        elif self.current_token[0] == "STATIC":
            return self.parse_static()
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
            return self.parse_return_statement()
        elif self.current_token[0] == "BREAK":
            return self.parse_break_statement()
        elif self.current_token[0] == "CONTINUE":
            return self.parse_continue_statement()
        else:
            expr = self.parse_expression()
            self.eat("SEMICOLON")
            return expr

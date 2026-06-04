"""Parser for Mojo source AST construction."""

import re

from .MojoAst import *
from .MojoLexer import *


class MojoParser:
    """Parse Mojo tokens into the Mojo backend AST."""

    STATEMENT_END_TOKENS = {"NEWLINE", "DEDENT", "EOF", "RBRACE"}
    ATTRIBUTE_START_TOKENS = {"ATTRIBUTE", "AT"}
    FUNCTION_TOKENS = {"FN", "DEF"}
    TYPE_START_TOKENS = {"IDENTIFIER", "INT", "FLOAT", "BOOL", "STRING"}
    FUNCTION_TYPE_TOKENS = {"FN", "DEF"}
    PARAMETER_CONVENTION_TOKENS = {"VAR"}
    PARAMETER_CONVENTION_IDENTIFIERS = {
        "borrowed",
        "inout",
        "mut",
        "owned",
        "out",
        "read",
        "ref",
        "deinit",
    }
    REFERENCE_TYPE_PREFIXES = {"ref"}
    FUNCTION_EFFECT_IDENTIFIERS = {"abi", "capturing", "raises", "unified"}

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.expression_layout_depth = 0
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def skip_newlines(self):
        while self.current_token[0] == "NEWLINE":
            self.eat("NEWLINE")

    def skip_layout_tokens(self):
        while self.current_token[0] in ["NEWLINE", "INDENT", "DEDENT"]:
            self.eat(self.current_token[0])

    def skip_expression_layout(self):
        if self.expression_layout_depth:
            self.skip_layout_tokens()

    def consume_statement_terminator(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return

        if self.current_token[0] in self.STATEMENT_END_TOKENS:
            return

        raise SyntaxError(f"Expected end of statement, got {self.current_token[0]}")

    def attach_attributes(self, node, attributes):
        if attributes:
            existing_attributes = getattr(node, "attributes", [])
            node.attributes = attributes + existing_attributes
        return node

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def peek_token(self, offset=1):
        index = self.pos + offset
        if index < len(self.tokens):
            return self.tokens[index]
        return ("EOF", None)

    def parse(self):
        module = self.parse_module()
        self.eat("EOF")
        return module

    def parse_module(self):
        imports = []
        structs = []
        functions = []
        global_variables = []
        constants = []
        classes = []
        traits = []
        all_items = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] in ["NEWLINE", "INDENT", "DEDENT"]:
                self.eat(self.current_token[0])
                continue

            attributes = self.parse_attributes()
            if attributes and self.current_token[0] == "EOF":
                raise SyntaxError("Expected declaration after attribute")

            if self.current_token[0] in ["IMPORT", "FROM"]:
                node = self.parse_import_statement()
                imports.append(node)
                all_items.append(node)
            elif self.current_token[0] == "STRUCT":
                node = self.parse_struct(attributes)
                structs.append(node)
                all_items.append(node)
            elif self.current_token[0] == "CLASS":
                node = self.parse_class(attributes)
                classes.append(node)
                all_items.append(node)
            elif self.current_token[0] == "TRAIT":
                node = self.parse_trait(attributes)
                traits.append(node)
                all_items.append(node)
            elif self.current_token[0] == "CONSTANT":
                node = self.parse_constant_buffer()
                self.attach_attributes(node, attributes)
                constants.append(node)
                all_items.append(node)
            elif self.current_token[0] in ["COMPTIME", "ALIAS"]:
                node = self.parse_comptime_or_alias_statement()
                self.attach_attributes(node, attributes)
                if isinstance(node, VariableDeclarationNode):
                    global_variables.append(node)
                all_items.append(node)
            elif self.current_token[0] in self.FUNCTION_TOKENS:
                node = self.parse_function(attributes)
                functions.append(node)
                all_items.append(node)
            elif self.current_token[0] in ["LET", "VAR"]:
                node = self.parse_variable_declaration_or_assignment()
                self.attach_attributes(node, attributes)
                global_variables.append(node)
                all_items.append(node)
            elif self.current_token[0] == "DECORATOR":
                node = self.parse_decorator()
                all_items.append(node)
            else:
                if attributes:
                    raise SyntaxError("Expected declaration after attribute")
                token_type, token_value = self.current_token
                raise SyntaxError(
                    f"Unexpected top-level token: {token_type} {token_value!r}"
                )

        return ShaderNode(
            includes=imports,
            functions=all_items,
            structs=structs,
            global_variables=global_variables,
            constants=constants,
            classes=classes,
            traits=traits,
        )

    def parse_import_statement(self):
        if self.current_token[0] == "FROM":
            self.eat("FROM")
            module_name = self.parse_import_path()
            self.eat("IMPORT")
            items = self.parse_import_items()
            self.consume_statement_terminator()
            self.skip_newlines()
            return ImportNode(module_name, items=items)

        self.eat("IMPORT")
        module_name = self.parse_import_path()
        alias = None
        if self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.current_token[1]
            self.eat("IDENTIFIER")
        self.consume_statement_terminator()
        self.skip_newlines()
        return ImportNode(module_name, alias=alias)

    def parse_import_path(self):
        name = ""
        while self.current_token[0] == "DOT":
            name += "."
            self.eat("DOT")

        if name and self.current_token[0] == "IMPORT":
            return name

        if self.is_identifier_like_token():
            name += self.current_token[1]
            self.eat(self.current_token[0])
        elif not name:
            raise SyntaxError(f"Expected import path, got {self.current_token[0]}")

        while self.current_token[0] == "DOT":
            self.eat("DOT")
            if not self.is_identifier_like_token():
                raise SyntaxError(
                    f"Expected IDENTIFIER after dot, got {self.current_token[0]}"
                )
            name += f".{self.current_token[1]}"
            self.eat(self.current_token[0])
        return name

    def is_identifier_like_token(self):
        token_value = self.current_token[1]
        return isinstance(token_value, str) and re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*$", token_value
        )

    def parse_import_items(self):
        items = []
        parenthesized = False
        self.skip_newlines()
        if self.current_token[0] == "LPAREN":
            parenthesized = True
            self.eat("LPAREN")
            self.skip_layout_tokens()

        while self.current_token[0] not in self.STATEMENT_END_TOKENS | {"SEMICOLON"}:
            if parenthesized:
                self.skip_layout_tokens()
            if parenthesized and self.current_token[0] == "RPAREN":
                break
            if self.current_token[0] == "MULTIPLY":
                item = self.current_token[1]
                self.eat("MULTIPLY")
            else:
                item = self.parse_import_path()

            if self.current_token[0] == "AS":
                self.eat("AS")
                alias = self.current_token[1]
                self.eat("IDENTIFIER")
                item = f"{item} as {alias}"

            items.append(item)
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if parenthesized:
                    self.skip_layout_tokens()
                else:
                    self.skip_newlines()
            else:
                break

        if parenthesized:
            self.skip_layout_tokens()
            self.eat("RPAREN")

        if not items:
            raise SyntaxError("Expected imported item")
        return items

    def parse_struct(self, initial_attributes=None):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        base_classes = []
        generic_parameters = None

        if self.current_token[0] == "LBRACKET":
            generic_parameters = self.parse_generic_type_suffix()

        if self.current_token[0] == "LPAREN":
            base_classes = self.parse_base_class_list()

        members = []
        methods = []

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                if self.current_token[0] in ["NEWLINE", "INDENT", "DEDENT"]:
                    self.eat(self.current_token[0])
                    continue
                self.add_struct_item(members, methods, self.parse_struct_member())
                self.skip_newlines()
            if self.current_token[0] == "RBRACE":
                self.eat("RBRACE")
        elif self.current_token[0] == "COLON":
            self.eat("COLON")
            self.skip_newlines()
            if self.current_token[0] == "INDENT":
                self.eat("INDENT")
            while (
                self.current_token[0] != "EOF"
                and self.current_token[0] != "DEDENT"
                and self.current_token[0] not in ["STRUCT", "CLASS"]
            ):
                self.skip_newlines()
                if self.current_token[0] in ["DEDENT", "EOF"]:
                    break
                self.add_struct_item(members, methods, self.parse_struct_member())
                self.skip_newlines()

            if self.current_token[0] == "DEDENT":
                self.eat("DEDENT")
        else:
            raise SyntaxError(f"Expected struct body, got {self.current_token[0]}")

        node = StructNode(name, members, attributes=initial_attributes)
        node.methods = methods
        node.base_classes = base_classes
        node.generic_parameters = generic_parameters
        return node

    def parse_base_class_list(self):
        base_classes = []
        self.eat("LPAREN")
        self.skip_layout_tokens()

        while self.current_token[0] != "RPAREN":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated base class list")
            base_classes.append(self.parse_type())
            self.skip_layout_tokens()

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                continue
            if self.current_token[0] != "RPAREN":
                raise SyntaxError(
                    f"Expected COMMA or RPAREN, got {self.current_token[0]}"
                )

        self.eat("RPAREN")
        return base_classes

    def parse_struct_member(self):
        self.skip_newlines()
        attributes = self.parse_attributes()
        if self.current_token[0] in self.FUNCTION_TOKENS:
            return self.parse_function(attributes)
        if self.current_token[0] in ["COMPTIME", "ALIAS"]:
            member = self.parse_comptime_or_alias_statement()
            return self.attach_attributes(member, attributes)
        return self.parse_typed_member("struct", attributes)

    def add_struct_item(self, members, methods, node):
        if isinstance(node, FunctionNode):
            methods.append(node)
        else:
            members.append(node)

    def parse_typed_member(self, context, initial_attributes=None):
        self.skip_newlines()
        member_attributes = list(initial_attributes or [])
        member_attributes.extend(self.parse_attributes())

        if self.current_token[0] in ["DEDENT", "EOF", "RBRACE"]:
            raise SyntaxError(f"Expected {context} member")

        if self.current_token[0] in ["LET", "VAR"]:
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                vtype = self.parse_type()
        elif self.peek_token()[0] == "COLON":
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("COLON")
            vtype = self.parse_type()
        else:
            if self.current_token[0] not in self.TYPE_START_TOKENS:
                if member_attributes:
                    raise SyntaxError(f"Expected {context} member after attribute")
                raise SyntaxError(
                    f"Unexpected token in {context}: {self.current_token[0]}"
                )
            vtype = self.parse_type()
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")

        member_attributes.extend(self.parse_attributes(skip_trailing_newlines=False))
        self.consume_statement_terminator()
        return VariableNode(vtype, var_name, attributes=member_attributes)

    def parse_class(self, initial_attributes=None):
        self.eat("CLASS")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        base_classes = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                base_classes.append(self.current_token[1])
                self.eat("IDENTIFIER")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
            self.eat("RPAREN")

        members = []
        methods = []

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                if self.current_token[0] in ["NEWLINE", "INDENT", "DEDENT"]:
                    self.eat(self.current_token[0])
                    continue
                self.add_class_member(members, methods, self.parse_class_member())
            if self.current_token[0] == "RBRACE":
                self.eat("RBRACE")
        elif self.current_token[0] == "COLON":
            self.eat("COLON")
            self.skip_newlines()
            if self.current_token[0] == "INDENT":
                self.eat("INDENT")
                while self.current_token[0] not in ["DEDENT", "EOF"]:
                    self.skip_newlines()
                    if self.current_token[0] in ["DEDENT", "EOF"]:
                        break
                    self.add_class_member(members, methods, self.parse_class_member())
                    self.skip_newlines()
                if self.current_token[0] == "DEDENT":
                    self.eat("DEDENT")
            else:
                self.add_class_member(members, methods, self.parse_class_member())
        else:
            raise SyntaxError(f"Expected class body, got {self.current_token[0]}")

        return ClassNode(
            name,
            members=members,
            methods=methods,
            base_classes=base_classes,
            attributes=initial_attributes,
        )

    def parse_trait(self, initial_attributes=None):
        self.eat("TRAIT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        base_classes = []
        generic_parameters = None

        if self.current_token[0] == "LBRACKET":
            generic_parameters = self.parse_generic_type_suffix()

        if self.current_token[0] == "LPAREN":
            base_classes = self.parse_base_class_list()

        members = []
        methods = []

        if self.current_token[0] == "COLON":
            self.eat("COLON")
            self.skip_newlines()
            if self.current_token[0] == "INDENT":
                self.eat("INDENT")
                while self.current_token[0] not in ["DEDENT", "EOF"]:
                    self.skip_newlines()
                    if self.current_token[0] in ["DEDENT", "EOF"]:
                        break
                    self.add_class_member(members, methods, self.parse_class_member())
                    self.skip_newlines()
                if self.current_token[0] == "DEDENT":
                    self.eat("DEDENT")
            else:
                self.add_class_member(members, methods, self.parse_class_member())
        else:
            raise SyntaxError(f"Expected trait body, got {self.current_token[0]}")

        node = TraitNode(
            name,
            members=members,
            methods=methods,
            base_classes=base_classes,
            attributes=initial_attributes,
        )
        node.generic_parameters = generic_parameters
        return node

    def parse_class_member(self):
        self.skip_newlines()
        attributes = self.parse_attributes()

        if self.is_ellipsis_statement():
            return self.parse_ellipsis_statement()
        if self.current_token[0] in self.FUNCTION_TOKENS:
            return self.parse_function(attributes)
        if self.current_token[0] == "CLASS":
            return self.parse_class(attributes)
        if self.current_token[0] in ["COMPTIME", "ALIAS"]:
            member = self.parse_comptime_or_alias_statement()
            return self.attach_attributes(member, attributes)
        if self.current_token[0] in ["LET", "VAR"]:
            member = self.parse_variable_declaration_or_assignment()
            return self.attach_attributes(member, attributes)
        if self.current_token[0] in self.TYPE_START_TOKENS:
            return self.parse_typed_member("class", attributes)

        if attributes:
            raise SyntaxError("Expected class member after attribute")
        raise SyntaxError(f"Unexpected token in class body: {self.current_token[0]}")

    def add_class_member(self, members, methods, node):
        if isinstance(node, FunctionNode):
            methods.append(node)
        else:
            members.append(node)

    def parse_constant_buffer(self):
        self.eat("CONSTANT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        members = []

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                if self.current_token[0] in ["NEWLINE", "INDENT", "DEDENT"]:
                    self.eat(self.current_token[0])
                    continue
                members.append(self.parse_constant_buffer_member())
            if self.current_token[0] == "RBRACE":
                self.eat("RBRACE")
        elif self.current_token[0] == "COLON":
            self.eat("COLON")
            self.skip_newlines()
            if self.current_token[0] == "INDENT":
                self.eat("INDENT")
                while self.current_token[0] not in ["DEDENT", "EOF"]:
                    self.skip_newlines()
                    if self.current_token[0] in ["DEDENT", "EOF"]:
                        break
                    members.append(self.parse_constant_buffer_member())
                    self.skip_newlines()
                if self.current_token[0] == "DEDENT":
                    self.eat("DEDENT")
            else:
                members.append(self.parse_constant_buffer_member())
        else:
            raise SyntaxError(
                f"Expected constant buffer body, got {self.current_token[0]}"
            )

        return ConstantBufferNode(name, members)

    def parse_constant_buffer_member(self):
        attributes = self.parse_attributes()

        if self.current_token[0] in ["LET", "VAR"]:
            member = self.parse_variable_declaration_or_assignment()
            return self.attach_attributes(member, attributes)

        if self.current_token[0] not in self.TYPE_START_TOKENS:
            if attributes:
                raise SyntaxError("Expected constant buffer member after attribute")
            raise SyntaxError(
                f"Unexpected token in constant buffer: {self.current_token[0]}"
            )

        if self.peek_token()[0] == "COLON":
            name = self.current_token[1]
            self.eat(self.current_token[0])
            self.eat("COLON")
            vtype = self.parse_type()
        else:
            vtype = self.parse_type()
            name = self.current_token[1]
            self.eat("IDENTIFIER")

        attributes.extend(self.parse_attributes(skip_trailing_newlines=False))
        self.consume_statement_terminator()
        return VariableNode(vtype, name, attributes=attributes)

    def parse_function(self, initial_attributes=None):
        attributes = list(initial_attributes or [])
        attributes.extend(self.parse_attributes())

        qualifier = None
        if self.current_token[0] in self.FUNCTION_TOKENS:
            self.eat(self.current_token[0])

        return_type = None
        if self.current_token[0] not in {"IDENTIFIER", "DEFAULT"}:
            raise SyntaxError(f"Expected function name, got {self.current_token[0]}")
        name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACKET":
            self.skip_bracketed_suffix()

        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")

        self.parse_function_signature_modifiers()
        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            if self.current_token[0] == "GREATER_THAN":
                self.eat("GREATER_THAN")
                return_type = self.parse_type()

        post_attributes = self.parse_attributes()
        attributes.extend(post_attributes)

        self.parse_function_signature_modifiers()

        where_clause = self.parse_where_clause()

        if self.current_token[0] == "COLON":
            self.eat("COLON")

        body = self.parse_block()

        func = FunctionNode(
            return_type, name, params, body, qualifiers=[], attributes=attributes
        )
        func.qualifier = qualifier
        func.where_clause = where_clause
        return func

    def parse_function_signature_modifiers(self):
        while True:
            previous_pos = self.pos
            self.parse_function_effects()
            self.parse_function_capture_list()
            if self.pos == previous_pos:
                return

    def parse_function_effects(self):
        while (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in self.FUNCTION_EFFECT_IDENTIFIERS
        ):
            self.eat("IDENTIFIER")
            self.skip_layout_tokens()
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_group("LPAREN", "RPAREN")
            elif self.current_token[0] == "LBRACE":
                self.skip_balanced_group("LBRACE", "RBRACE")
            self.skip_layout_tokens()

    def parse_function_capture_list(self):
        if not self.is_function_capture_list():
            return

        self.skip_balanced_group("LBRACE", "RBRACE")
        self.skip_layout_tokens()

    def is_function_capture_list(self):
        if self.current_token[0] != "LBRACE":
            return False

        index = self.pos
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                if depth == 0:
                    index += 1
                    while index < len(self.tokens) and self.tokens[index][0] in {
                        "NEWLINE",
                        "INDENT",
                        "DEDENT",
                    }:
                        index += 1
                    if index >= len(self.tokens):
                        return False
                    next_type, next_value = self.tokens[index]
                    if next_type in {"COLON", "MINUS"}:
                        return True
                    return (
                        next_type == "IDENTIFIER"
                        and next_value in self.FUNCTION_EFFECT_IDENTIFIERS | {"where"}
                    )
            index += 1

        return False

    def skip_balanced_group(self, open_token, close_token):
        self.eat(open_token)
        depth = 1
        while depth > 0:
            if self.current_token[0] == "EOF":
                raise SyntaxError(f"Unterminated Mojo {open_token} group")
            token_type = self.current_token[0]
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
            self.eat(token_type)

    def parse_where_clause(self):
        if not (
            self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "where"
        ):
            return None

        self.eat("IDENTIFIER")
        self.skip_layout_tokens()

        if self.current_token[0] != "LPAREN":
            raise SyntaxError("Expected parenthesized Mojo where clause")

        parts = []
        depth = 0
        while True:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated Mojo where clause")

            token_type, token_value = self.current_token
            if token_type in {"NEWLINE", "INDENT", "DEDENT"}:
                self.eat(token_type)
                continue

            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1

            parts.append(str(token_value))
            self.eat(token_type)

            if depth == 0:
                break

        return " ".join(parts)

    def parse_parameters(self):
        params = []
        while True:
            self.skip_layout_tokens()
            if self.current_token[0] == "RPAREN":
                break

            if self.consume_parameter_separator_marker():
                continue

            attributes = self.parse_attributes()
            self.skip_layout_tokens()
            convention = self.parse_parameter_convention()
            is_variadic = self.parse_variadic_parameter_marker()

            if self.is_untyped_self_parameter(convention):
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                vtype = ""
            elif self.is_bare_self_parameter():
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                vtype = ""
            elif self.current_token[0] in self.TYPE_START_TOKENS:
                if self.peek_token()[0] == "COLON":
                    name = self.current_token[1]
                    self.eat(self.current_token[0])
                    self.eat("COLON")
                    vtype = self.parse_type()
                else:
                    vtype = self.parse_type()
                    name = ""
                    if self.current_token[0] == "IDENTIFIER":
                        name = self.current_token[1]
                        self.eat("IDENTIFIER")

            else:
                raise SyntaxError(
                    f"Unexpected token in parameter list: {self.current_token[0]}"
                )

            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()

            param_attributes = self.parse_attributes(skip_trailing_newlines=False)
            attributes.extend(param_attributes)
            param = VariableNode(
                vtype,
                name,
                attributes=attributes,
                parameter_convention=convention,
            )
            param.default_value = default_value
            param.is_variadic = is_variadic
            params.append(param)
            self.skip_layout_tokens()

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] == "RPAREN":
                    break
            elif self.current_token[0] == "RPAREN":
                break
            else:
                raise SyntaxError(
                    f"Expected comma or closing parenthesis, got {self.current_token[0]}"
                )

        return params

    def consume_parameter_separator_marker(self):
        if self.current_token[0] not in {"MULTIPLY", "DIVIDE"}:
            return False

        next_token = self.peek_token()[0]
        if next_token not in {"COMMA", "RPAREN"}:
            return False

        self.eat(self.current_token[0])
        self.skip_layout_tokens()
        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
        return True

    def parse_parameter_convention(self):
        token_type, token_value = self.current_token
        if token_type in self.PARAMETER_CONVENTION_TOKENS:
            convention = token_value
        elif (
            token_type == "IDENTIFIER"
            and token_value in self.PARAMETER_CONVENTION_IDENTIFIERS
        ):
            convention = token_value
        else:
            return None

        if not self.is_parameter_convention_followed_by_parameter():
            return None

        self.eat(token_type)
        if self.current_token[0] == "LBRACKET":
            convention += self.parse_generic_type_suffix()
        self.skip_layout_tokens()
        return convention

    def is_parameter_convention_followed_by_parameter(self):
        next_token = self.peek_token()
        if next_token[0] in self.TYPE_START_TOKENS:
            return True
        if next_token[0] == "LBRACKET":
            return self.is_bracketed_parameter_convention_followed_by_parameter()
        if next_token[0] == "MULTIPLY":
            return self.peek_token(2)[0] in self.TYPE_START_TOKENS
        return False

    def is_bracketed_parameter_convention_followed_by_parameter(self):
        index = self.pos + 1
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    index += 1
                    while index < len(self.tokens) and self.tokens[index][0] in {
                        "NEWLINE",
                        "INDENT",
                        "DEDENT",
                    }:
                        index += 1
                    if index >= len(self.tokens):
                        return False
                    return self.tokens[index][0] in self.TYPE_START_TOKENS
            index += 1

        return False

    def parse_variadic_parameter_marker(self):
        if self.current_token[0] != "MULTIPLY":
            return False
        if self.peek_token()[0] not in self.TYPE_START_TOKENS:
            return False
        self.eat("MULTIPLY")
        self.skip_layout_tokens()
        return True

    def is_untyped_self_parameter(self, convention):
        if convention is None:
            return False
        if self.current_token[0] != "IDENTIFIER" or self.current_token[1] != "self":
            return False
        return self.peek_token()[0] in {"COMMA", "RPAREN"}

    def is_bare_self_parameter(self):
        if self.current_token[0] != "IDENTIFIER" or self.current_token[1] != "self":
            return False
        return self.peek_token()[0] in {"COMMA", "RPAREN"}

    def parse_attributes(self, skip_trailing_newlines=True):
        attributes = []
        while self.current_token[0] in self.ATTRIBUTE_START_TOKENS:
            if self.current_token[0] == "ATTRIBUTE":
                attr_content = self.current_token[1][2:-2]  # Remove [[ and ]]
                attr_parts = attr_content.split("(")
                if len(attr_parts) > 1:
                    name = attr_parts[0]
                    args = attr_parts[1][:-1].split(",")
                else:
                    name = attr_content
                    args = []
                attributes.append(AttributeNode(name, args))
                self.eat("ATTRIBUTE")
            else:
                self.eat("AT")
                if self.current_token[0] != "IDENTIFIER":
                    raise SyntaxError(
                        f"Expected attribute name, got {self.current_token[0]}"
                    )
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                while self.current_token[0] == "DOT":
                    self.eat("DOT")
                    if self.current_token[0] != "IDENTIFIER":
                        raise SyntaxError(
                            f"Expected attribute name after dot, got {self.current_token[0]}"
                        )
                    name += f".{self.current_token[1]}"
                    self.eat("IDENTIFIER")
                args = []
                if self.current_token[0] == "LPAREN":
                    self.eat("LPAREN")
                    self.skip_layout_tokens()
                    while self.current_token[0] != "RPAREN":
                        self.expression_layout_depth += 1
                        try:
                            args.append(self.parse_expression())
                        finally:
                            self.expression_layout_depth -= 1
                        self.skip_layout_tokens()
                        if self.current_token[0] == "COMMA":
                            self.eat("COMMA")
                            self.skip_layout_tokens()
                        elif self.current_token[0] != "RPAREN":
                            raise SyntaxError(
                                f"Expected COMMA or RPAREN, got {self.current_token[0]}"
                            )
                    self.eat("RPAREN")
                attributes.append(AttributeNode(name, args))
            if skip_trailing_newlines:
                self.skip_newlines()
        return attributes

    def parse_block(self):
        self.skip_newlines()
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            statements = []
            while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
                if self.current_token[0] == "NEWLINE":
                    self.eat("NEWLINE")
                    continue
                statements.append(self.parse_statement())
                self.skip_newlines()
            if self.current_token[0] == "RBRACE":
                self.eat("RBRACE")
            return statements
        elif self.current_token[0] == "INDENT":
            self.eat("INDENT")
            statements = []
            while self.current_token[0] not in ["DEDENT", "EOF"]:
                self.skip_newlines()
                if self.current_token[0] in ["DEDENT", "EOF"]:
                    break
                statements.append(self.parse_statement())
                self.skip_newlines()
            if self.current_token[0] == "DEDENT":
                self.eat("DEDENT")
            return statements
        else:
            statements = []
            if self.current_token[0] not in [
                "DEDENT",
                "EOF",
                "ELSE",
                "ELIF",
                "CASE",
                "DEFAULT",
            ]:
                statements.append(self.parse_statement())
            return statements

    def parse_statement(self):
        self.skip_newlines()
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "try":
            return self.parse_try_except_statement()
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "assert":
            return self.parse_assert_statement()
        if self.current_token[0] in ["IMPORT", "FROM"]:
            return self.parse_import_statement()

        if self.current_token[0] in [
            "FLOAT",
            "INT",
            "UINT",
            "BOOL",
            "IDENTIFIER",
            "LET",
            "VAR",
            "COMPTIME",
            "ALIAS",
        ]:
            return self.parse_variable_declaration_or_assignment()

        elif self.current_token[0] in self.ATTRIBUTE_START_TOKENS:
            attributes = self.parse_attributes()
            if self.current_token[0] in self.FUNCTION_TOKENS:
                return self.parse_function(attributes)
            if self.current_token[0] == "STRUCT":
                return self.parse_struct(attributes)
            if self.current_token[0] == "CLASS":
                return self.parse_class(attributes)
            statement = self.parse_statement()
            if attributes:
                existing_attributes = getattr(statement, "attributes", [])
                statement.attributes = attributes + existing_attributes
            return statement

        elif self.current_token[0] in self.FUNCTION_TOKENS:
            return self.parse_function()
        elif self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
        elif self.current_token[0] == "WITH":
            return self.parse_with_statement()
        elif self.current_token[0] == "RETURN":
            return self.parse_return_statement()
        elif self.current_token[0] == "RAISE":
            return self.parse_raise_statement()
        elif self.current_token[0] == "BREAK":
            self.eat("BREAK")
            self.consume_statement_terminator()
            return BreakNode()
        elif self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            self.consume_statement_terminator()
            return ContinueNode()
        elif self.current_token[0] == "PASS":
            self.eat("PASS")
            self.consume_statement_terminator()
            return PassNode()
        elif self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()
        elif self.current_token[0] == "STRUCT":
            return self.parse_struct()
        elif self.current_token[0] == "TRAIT":
            return self.parse_trait()
        elif self.is_ellipsis_statement():
            return self.parse_ellipsis_statement()
        else:
            return self.parse_expression_statement()

    def is_ellipsis_statement(self):
        return (
            self.current_token[0] == "DOT"
            and self.peek_token()[0] == "DOT"
            and self.peek_token(2)[0] == "DOT"
        )

    def parse_ellipsis_statement(self):
        self.eat("DOT")
        self.eat("DOT")
        self.eat("DOT")
        self.consume_statement_terminator()
        return PassNode()

    def parse_variable_declaration_or_assignment(self):
        if self.current_token[0] in ["COMPTIME", "ALIAS"]:
            return self.parse_comptime_or_alias_statement()

        if self.current_token[0] in ["LET", "VAR"]:
            var_type = self.current_token[0]
            self.eat(self.current_token[0])
            name = self.parse_identifier_tuple()

            attributes = []
            vtype = None
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                vtype = self.parse_type()
                attributes.extend(self.parse_attributes(skip_trailing_newlines=False))

            initial_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                initial_value = self.parse_expression_list_value()
                attributes.extend(self.parse_attributes(skip_trailing_newlines=False))

            self.consume_statement_terminator()

            return VariableDeclarationNode(
                vtype,
                name,
                initial_value,
                is_var=var_type == "VAR",
                attributes=attributes,
            )

        if self.current_token[0] == "IDENTIFIER" and self.peek_token()[0] == "COMMA":
            left = self.parse_identifier_tuple()
            if self.current_token[0] not in [
                "EQUALS",
                "PLUS_EQUALS",
                "MINUS_EQUALS",
                "MULTIPLY_EQUALS",
                "DIVIDE_EQUALS",
                "ASSIGN_XOR",
                "ASSIGN_OR",
                "ASSIGN_AND",
                "ASSIGN_SHIFT_LEFT",
                "ASSIGN_SHIFT_RIGHT",
                "ASSIGN_MOD",
            ]:
                raise SyntaxError("Expected assignment after identifier tuple")
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_expression_list_value()
            self.consume_statement_terminator()
            return AssignmentNode(left, right, op)

        if self.current_token[0] == "IDENTIFIER" and self.peek_token()[0] == "COLON":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("COLON")
            vtype = self.parse_type()
            attributes = self.parse_attributes(skip_trailing_newlines=False)
            initial_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                initial_value = self.parse_expression()
                attributes.extend(self.parse_attributes(skip_trailing_newlines=False))
            self.consume_statement_terminator()
            if initial_value is not None:
                return VariableDeclarationNode(
                    vtype,
                    name,
                    initial_value,
                    is_var=True,
                    attributes=attributes,
                )
            return VariableNode(vtype, name, attributes=attributes)

        statement = self.parse_assignment()
        self.consume_statement_terminator()
        return statement

    def parse_identifier_tuple(self):
        identifiers = [VariableNode("", self.current_token[1])]
        self.eat("IDENTIFIER")

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected IDENTIFIER after comma, got {self.current_token[0]}"
                )
            identifiers.append(VariableNode("", self.current_token[1]))
            self.eat("IDENTIFIER")

        if len(identifiers) == 1:
            return identifiers[0].name
        return TupleNode(identifiers)

    def parse_expression_list_value(self):
        values = [self.parse_expression()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if self.current_token[0] in self.STATEMENT_END_TOKENS | {"SEMICOLON"}:
                break
            values.append(self.parse_expression())

        if len(values) == 1:
            return values[0]
        return TupleNode(values)

    def parse_comptime_or_alias_statement(self):
        if self.current_token[0] == "ALIAS":
            return self.parse_alias_declaration()
        return self.parse_comptime_statement()

    def parse_alias_declaration(self):
        self.eat("ALIAS")
        node = self.parse_comptime_declaration(after_keyword=True)
        node.is_alias = True
        return node

    def parse_comptime_statement(self):
        self.eat("COMPTIME")
        if self.current_token[0] == "IF":
            node = self.parse_if_statement()
            node.is_comptime = True
            return node
        if self.current_token[0] == "FOR":
            node = self.parse_for_statement()
            node.is_comptime = True
            return node
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "assert":
            return self.parse_assert_statement(is_comptime=True)
        if self.is_comptime_declaration_start():
            return self.parse_comptime_declaration(after_keyword=True)
        if self.is_comptime_expression_statement():
            node = self.parse_expression()
            self.consume_statement_terminator()
            node.is_comptime = True
            return node
        return self.parse_comptime_declaration(after_keyword=True)

    def is_comptime_declaration_start(self):
        if self.current_token[0] != "IDENTIFIER":
            return False
        next_token = self.peek_token()[0]
        if next_token in {"COLON", "EQUALS"}:
            return True
        if next_token != "LBRACKET":
            return False
        index = self.pos + 1
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    next_after_suffix = (
                        self.tokens[index + 1]
                        if index + 1 < len(self.tokens)
                        else ("EOF", None)
                    )
                    return next_after_suffix[0] in {"COLON", "EQUALS"}
            index += 1
        return False

    def parse_assert_statement(self, is_comptime=False):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            node = self.parse_call(VariableNode("", "assert"))
            args = node.args
        else:
            args = [self.parse_expression()]

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            self.skip_layout_tokens()
            args.append(self.parse_expression())

        self.consume_statement_terminator()
        node = FunctionCallNode("assert", args)
        if is_comptime:
            node.is_comptime = True
        return node

    def is_comptime_expression_statement(self):
        if self.current_token[0] != "IDENTIFIER":
            return False
        return self.peek_token()[0] not in {
            "COLON",
            "EQUALS",
            "SEMICOLON",
            "NEWLINE",
            "DEDENT",
            "EOF",
            "RBRACE",
        }

    def parse_comptime_declaration(self, after_keyword=False):
        if not after_keyword:
            self.eat("COMPTIME")

        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LBRACKET":
            name += self.parse_generic_type_suffix()

        vtype = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            vtype = self.parse_type()

        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()

        self.consume_statement_terminator()
        node = VariableDeclarationNode(vtype, name, initial_value, is_var=False)
        node.is_comptime = True
        return node

    def parse_if_statement(self):
        self.eat("IF")
        condition = self.parse_expression()
        self.eat("COLON")
        if_body = self.parse_block()
        self.skip_newlines()

        else_body = None
        while self.current_token[0] == "ELIF":
            self.eat("ELIF")
            elif_condition = self.parse_expression()
            self.eat("COLON")
            elif_body = self.parse_block()
            self.skip_newlines()
            elif_node = IfNode(elif_condition, elif_body, None)
            if else_body is None:
                else_body = [elif_node]
            else:
                current = else_body[0]
                while isinstance(current, IfNode) and current.else_body:
                    current = current.else_body[0]
                current.else_body = [elif_node]

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            self.eat("COLON")
            final_else_body = self.parse_block()
            if else_body is None:
                else_body = final_else_body
            else:
                current = else_body[0]
                while isinstance(current, IfNode) and current.else_body:
                    current = current.else_body[0]
                current.else_body = final_else_body

        return IfNode(condition, if_body, else_body)

    def parse_for_statement(self):
        self.eat("FOR")

        saved_pos = self.pos
        saved_token = self.current_token

        try:
            init = self.parse_variable_declaration_or_assignment()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

            condition = self.parse_expression()
            self.eat("SEMICOLON")
            update = self.parse_expression()
            self.eat("COLON")
            body = self.parse_block()
            return ForNode(init, condition, update, body)

        except Exception:
            self.pos = saved_pos
            self.current_token = saved_token

        if self.current_token[0] == "IDENTIFIER":
            var_name = self.parse_identifier_tuple()
            if self.current_token[0] == "IN":
                self.eat("IN")
                iterable = self.parse_for_iterable()
                self.eat("COLON")
                body = self.parse_block()
                return RangeForNode("", var_name, iterable, body)

        raise SyntaxError(
            f"Invalid for loop syntax. Expected C-style 'for init; condition; update:' or Python-style 'for var in iterable:'"
        )

    def parse_for_iterable(self):
        if self.current_token[0] == "IDENTIFIER" and self.peek_token()[0] == "COLON":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return VariableNode("", name)

        return self.parse_expression()

    def parse_while_statement(self):
        self.eat("WHILE")
        condition = self.parse_expression()
        self.eat("COLON")
        body = self.parse_block()
        return WhileNode(condition, body)

    def parse_with_statement(self):
        self.eat("WITH")
        context_expr = self.parse_expression()
        alias = None
        if isinstance(context_expr, CastNode) and self.current_token[0] == "COLON":
            alias = context_expr.target_type
            context_expr = context_expr.expression
        elif self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.current_token[1]
            self.eat("IDENTIFIER")
        self.eat("COLON")
        body = self.parse_block()
        return WithNode(context_expr, alias, body)

    def parse_try_except_statement(self):
        self.eat("IDENTIFIER")
        self.eat("COLON")
        try_body = self.parse_block()
        self.skip_newlines()

        except_body = []
        exception_name = None
        else_body = []
        finally_body = []

        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "except":
            self.eat("IDENTIFIER")
            if self.current_token[0] == "IDENTIFIER":
                exception_name = self.current_token[1]
                self.eat("IDENTIFIER")
            self.eat("COLON")
            except_body = self.parse_block()
            self.skip_newlines()

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            self.eat("COLON")
            else_body = self.parse_block()
            self.skip_newlines()

        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "finally":
            self.eat("IDENTIFIER")
            self.eat("COLON")
            finally_body = self.parse_block()
            self.skip_newlines()

        if not except_body and not finally_body:
            raise SyntaxError("Expected except or finally clause after try block")

        return TryExceptNode(
            try_body,
            except_body,
            exception_name,
            else_body=else_body,
            finally_body=finally_body,
        )

    def parse_switch_statement(self):
        self.eat("SWITCH")
        expression = self.parse_expression()
        self.eat("COLON")
        cases = []
        seen_default = False
        self.skip_newlines()
        if self.current_token[0] == "INDENT":
            self.eat("INDENT")
            self.skip_newlines()
            while self.current_token[0] in ["CASE", "DEFAULT"]:
                if self.current_token[0] == "DEFAULT":
                    if seen_default:
                        raise SyntaxError("duplicate default label in switch")
                    seen_default = True
                cases.append(self.parse_switch_case())
                self.skip_newlines()
            if self.current_token[0] == "DEDENT":
                self.eat("DEDENT")
        else:
            while self.current_token[0] in ["CASE", "DEFAULT"]:
                if self.current_token[0] == "DEFAULT":
                    if seen_default:
                        raise SyntaxError("duplicate default label in switch")
                    seen_default = True
                cases.append(self.parse_switch_case())
                self.skip_newlines()
        return SwitchNode(expression, cases)

    def parse_switch_case(self):
        if self.current_token[0] == "CASE":
            self.eat("CASE")
            condition = self.parse_expression()
            self.eat("COLON")
            body = self.parse_block()
            return SwitchCaseNode(condition, body)
        elif self.current_token[0] == "DEFAULT":
            self.eat("DEFAULT")
            self.eat("COLON")
            body = self.parse_block()
            return SwitchCaseNode(None, body)
        else:
            raise SyntaxError(
                f"Unexpected token in switch case: {self.current_token[0]}"
            )

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] in [
            "SEMICOLON",
            "NEWLINE",
            "DEDENT",
            "EOF",
            "RBRACE",
        ]:
            value = None
        else:
            value = self.parse_expression()

        self.consume_statement_terminator()

        return ReturnNode(value)

    def parse_raise_statement(self):
        self.eat("RAISE")
        if self.current_token[0] in self.STATEMENT_END_TOKENS | {"SEMICOLON"}:
            value = None
        else:
            value = self.parse_expression()

        self.consume_statement_terminator()
        args = [] if value is None else [value]
        return FunctionCallNode("raise", args)

    def parse_expression_statement(self):
        expr = self.parse_expression()
        self.consume_statement_terminator()
        return expr

    def parse_expression(self):
        return self.parse_assignment()

    def parse_assignment(self):
        left = self.parse_logical_or()
        self.skip_expression_layout()
        if self.current_token[0] in [
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
            "ASSIGN_XOR",
            "ASSIGN_OR",
            "ASSIGN_AND",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
            "ASSIGN_MOD",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment()
            return AssignmentNode(left, right, op)
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)
        self.skip_expression_layout()
        if self.current_token[0] == "IF" and self.is_inline_if_expression():
            true_expr = left
            self.eat("IF")
            condition = self.parse_expression()
            self.eat("ELSE")
            false_expr = self.parse_expression()
            left = TernaryOpNode(condition, true_expr, false_expr)
        return left

    def is_inline_if_expression(self):
        depth = 0
        idx = self.pos
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "RBRACE"}:
                if depth == 0:
                    return False
                depth -= 1
            elif depth == 0:
                if token_type == "ELSE":
                    return True
                if token_type in {"NEWLINE", "DEDENT", "EOF", "COMMA", "SEMICOLON"}:
                    return False
            idx += 1
        return False

    def parse_logical_or(self):
        left = self.parse_logical_and()
        self.skip_expression_layout()
        while self.current_token[0] == "OR":
            op = self.current_token[1]
            self.eat("OR")
            right = self.parse_logical_and()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_logical_and(self):
        left = self.parse_bitwise_or()
        self.skip_expression_layout()
        while self.current_token[0] == "AND":
            op = self.current_token[1]
            self.eat("AND")
            right = self.parse_bitwise_or()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_bitwise_or(self):
        left = self.parse_bitwise_xor()
        self.skip_expression_layout()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_bitwise_xor(self):
        left = self.parse_bitwise_and()
        self.skip_expression_layout()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_bitwise_and(self):
        left = self.parse_equality()
        self.skip_expression_layout()
        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_equality(self):
        left = self.parse_relational()
        self.skip_expression_layout()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_relational(self):
        left = self.parse_shift()
        self.skip_expression_layout()
        comparisons = []
        while (
            self.current_token[0]
            in [
                "LESS_THAN",
                "GREATER_THAN",
                "LESS_EQUAL",
                "GREATER_EQUAL",
                "IN",
            ]
            or self.is_identity_operator()
        ):
            if self.is_identity_operator():
                op = self.parse_identity_operator()
            else:
                op = self.current_token[1]
                self.eat(self.current_token[0])
            right = self.parse_shift()
            comparisons.append((left, op, right))
            left = right
            self.skip_expression_layout()
        if not comparisons:
            return left

        left, op, right = comparisons[0]
        expression = BinaryOpNode(left, op, right)
        for left, op, right in comparisons[1:]:
            expression = BinaryOpNode(
                expression,
                "&&",
                BinaryOpNode(left, op, right),
            )
        return expression

    def is_identity_operator(self):
        return self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "is"

    def parse_identity_operator(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "NOT":
            self.eat("NOT")
            return "is not"
        return "is"

    def parse_shift(self):
        left = self.parse_additive()
        self.skip_expression_layout()
        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        self.skip_expression_layout()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        self.skip_expression_layout()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "FLOOR_DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_unary(self):
        if self.current_token[0] == "COMPTIME":
            self.eat("COMPTIME")
            operand = self.parse_unary()
            operand.is_comptime = True
            return operand
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT", "NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_power()

    def parse_power(self):
        left = self.parse_primary()
        if self.current_token[0] == "POWER":
            op = self.current_token[1]
            self.eat("POWER")
            right = self.parse_unary()
            return BinaryOpNode(left, op, right)
        return left

    def parse_primary(self):
        if self.current_token[0] in {"IDENTIFIER", "DEFAULT"}:
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "BACKTICK_IDENTIFIER":
            value = self.current_token[1]
            self.eat("BACKTICK_IDENTIFIER")
            return self.parse_postfix_suffixes(value)
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return self.parse_postfix_suffixes(value)
        elif self.current_token[0] == "STRING_LITERAL":
            value = self.current_token[1]
            self.eat("STRING_LITERAL")
            return self.parse_postfix_suffixes(value)
        elif self.current_token[0] == "BOOL_LITERAL":
            value = self.current_token[1]
            self.eat("BOOL_LITERAL")
            return self.parse_postfix_suffixes(value)
        elif self.current_token[0] in ["INT", "FLOAT", "BOOL", "STRING"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[0] == "IDENTIFIER":
                var_name = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "COLON":
                    self.eat("COLON")
                    type_annotation = self.current_token[1]
                    self.eat(self.current_token[0])
                    return VariableNode(type_annotation, var_name)
                return VariableNode(type_name, var_name)
            elif self.current_token[0] == "LPAREN":
                return self.parse_postfix_suffixes(
                    self.parse_vector_constructor(type_name)
                )
            elif self.current_token[0] in {"DOT", "LBRACKET", "AS"}:
                return self.parse_postfix_suffixes(VariableNode("", type_name))
            return type_name
        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            self.skip_layout_tokens()
            self.expression_layout_depth += 1
            try:
                expr = self.parse_expression()
            finally:
                self.expression_layout_depth -= 1
            self.skip_layout_tokens()
            expr = self.parse_adjacent_string_literals(expr)
            self.skip_layout_tokens()
            if self.current_token[0] == "COMMA":
                elements = [expr]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    self.skip_layout_tokens()
                    if self.current_token[0] == "RPAREN":
                        break
                    elements.append(self.parse_expression())
                    self.skip_layout_tokens()
                self.eat("RPAREN")
                return self.parse_postfix_suffixes(TupleNode(elements))
            self.eat("RPAREN")
            return self.parse_postfix_suffixes(expr)
        elif self.current_token[0] == "LBRACKET":
            return self.parse_list_literal()
        elif self.current_token[0] in ["FN", "DEF", "STRUCT", "CLASS", "LET", "VAR"]:
            raise SyntaxError(f"Unexpected top-level keyword: {self.current_token[0]}")
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def parse_list_literal(self):
        self.eat("LBRACKET")
        elements = []
        self.skip_layout_tokens()

        while self.current_token[0] != "RBRACKET":
            elements.append(self.parse_expression())
            self.skip_layout_tokens()
            if len(elements) == 1 and self.current_token[0] == "FOR":
                comprehension = self.parse_list_comprehension(elements[0])
                self.eat("RBRACKET")
                return self.parse_postfix_suffixes(comprehension)
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] == "RBRACKET":
                    break
            elif self.current_token[0] != "RBRACKET":
                raise SyntaxError(
                    f"Expected COMMA or RBRACKET, got {self.current_token[0]}"
                )

        self.eat("RBRACKET")
        return self.parse_postfix_suffixes(ListLiteralNode(elements))

    def parse_list_comprehension(self, expression):
        clauses = []
        while self.current_token[0] == "FOR":
            self.eat("FOR")
            pattern = self.parse_comprehension_pattern()
            self.eat("IN")
            iterable = self.parse_expression()
            clauses.append({"kind": "for", "pattern": pattern, "iterable": iterable})
            self.skip_layout_tokens()

            while self.current_token[0] == "IF":
                self.eat("IF")
                condition = self.parse_expression()
                clauses.append({"kind": "if", "condition": condition})
                self.skip_layout_tokens()

        return ListComprehensionNode(expression, clauses)

    def parse_comprehension_pattern(self):
        if not self.is_identifier_like_token():
            raise SyntaxError(
                f"Expected comprehension target, got {self.current_token[0]}"
            )

        names = [self.current_token[1]]
        self.eat(self.current_token[0])
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if not self.is_identifier_like_token():
                raise SyntaxError(
                    f"Expected comprehension target, got {self.current_token[0]}"
                )
            names.append(self.current_token[1])
            self.eat(self.current_token[0])

        if len(names) == 1:
            return names[0]
        return TupleNode(names)

    def parse_vector_constructor(self, type_name):
        self.eat("LPAREN")
        args = []
        self.skip_layout_tokens()
        while self.current_token[0] != "RPAREN":
            self.expression_layout_depth += 1
            try:
                arg = self.parse_expression()
            finally:
                self.expression_layout_depth -= 1
            self.skip_layout_tokens()
            arg = self.parse_adjacent_string_literals(arg)
            args.append(arg)
            self.skip_layout_tokens()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
            elif self.current_token[0] != "RPAREN":
                raise SyntaxError(
                    f"Expected COMMA or RPAREN, got {self.current_token[0]}"
                )
        self.eat("RPAREN")
        return VectorConstructorNode(type_name, args)

    def parse_function_call_or_identifier(self):
        if self.current_token[0] in {"IDENTIFIER", "DEFAULT"}:
            name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[0] == "STRING_LITERAL":
                value = f"{name}{self.current_token[1]}"
                self.eat("STRING_LITERAL")
                return self.parse_postfix_suffixes(value)
            return self.parse_postfix_suffixes(VariableNode("", name))

        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value

        else:
            raise SyntaxError(
                f"Expected IDENTIFIER or NUMBER, got {self.current_token[0]}"
            )

    def parse_postfix_suffixes(self, node):
        while True:
            self.skip_expression_layout()

            if self.current_token[0] == "LPAREN":
                node = self.parse_call(node)
                continue

            if self.current_token[0] == "DOT":
                self.eat("DOT")
                if not self.is_identifier_like_token():
                    raise SyntaxError(
                        f"Expected IDENTIFIER after dot, got {self.current_token[0]}"
                    )
                member = self.current_token[1]
                self.eat(self.current_token[0])
                node = MemberAccessNode(node, member)
                continue

            if self.current_token[0] == "LBRACKET":
                if self.is_generic_constructor_suffix(node):
                    type_name = node.name + self.parse_generic_type_suffix()
                    node = self.parse_vector_constructor(type_name)
                else:
                    node = self.parse_array_access(node)
                continue

            if self.current_token[0] == "AS":
                self.eat("AS")
                node = CastNode(self.parse_type(), node)
                continue

            if self.current_token[0] == "BITWISE_XOR":
                if not self.is_postfix_transfer_marker():
                    return node
                self.eat("BITWISE_XOR")
                node.is_transfer = True
                continue

            return node

    def is_postfix_transfer_marker(self):
        return self.peek_token()[0] in self.STATEMENT_END_TOKENS | {
            "COMMA",
            "RPAREN",
            "RBRACKET",
            "COLON",
        }

    def parse_call(self, callee):
        self.eat("LPAREN")
        args = []
        self.skip_layout_tokens()
        while self.current_token[0] != "RPAREN":
            self.expression_layout_depth += 1
            try:
                arg = self.parse_expression()
            finally:
                self.expression_layout_depth -= 1
            self.skip_layout_tokens()
            arg = self.parse_adjacent_string_literals(arg)
            args.append(arg)
            self.skip_layout_tokens()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
            elif self.current_token[0] != "RPAREN":
                raise SyntaxError(
                    f"Expected COMMA or RPAREN, got {self.current_token[0]}"
                )
        self.eat("RPAREN")
        if isinstance(callee, VariableNode):
            return FunctionCallNode(callee.name, args)
        if isinstance(callee, MemberAccessNode):
            return MethodCallNode(callee.object, callee.member, args)
        return CallNode(callee, args)

    def parse_adjacent_string_literals(self, arg):
        if not self.is_string_literal_value(arg):
            return arg

        while self.current_token[0] == "STRING_LITERAL":
            next_literal = self.current_token[1]
            self.eat("STRING_LITERAL")
            arg = self.concat_string_literals(arg, next_literal)
            self.skip_layout_tokens()
        return arg

    def is_string_literal_value(self, value):
        return isinstance(value, str) and len(value) >= 2 and value[0] in {"'", '"'}

    def concat_string_literals(self, left, right):
        quote = left[0]
        return f"{quote}{left[1:-1]}{right[1:-1]}{quote}"

    def parse_decorator(self):
        self.eat("DECORATOR")
        name = self.current_token[1]
        args = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                args.append(self.parse_expression())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
            self.eat("RPAREN")
        return DecoratorNode(name, args)

    def parse_array_access(self, array):
        self.eat("LBRACKET")
        self.skip_layout_tokens()
        if self.current_token[0] == "RBRACKET":
            self.eat("RBRACKET")
            return ArrayAccessNode(array, TupleNode([]))

        self.expression_layout_depth += 1
        try:
            indices = [self.parse_index_component()]
        finally:
            self.expression_layout_depth -= 1
        self.skip_layout_tokens()
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            self.skip_layout_tokens()
            if self.current_token[0] == "RBRACKET":
                break
            self.expression_layout_depth += 1
            try:
                indices.append(self.parse_index_component())
            finally:
                self.expression_layout_depth -= 1
            self.skip_layout_tokens()
        self.eat("RBRACKET")
        index = indices[0] if len(indices) == 1 else TupleNode(indices)
        return ArrayAccessNode(array, index)

    def parse_index_component(self):
        if self.current_token[0] == "COLON":
            return self.parse_slice_index(None)

        value = self.parse_expression()
        self.skip_expression_layout()
        if self.current_token[0] == "COLON":
            return self.parse_slice_index(value)
        return value

    def parse_slice_index(self, start):
        self.eat("COLON")
        self.skip_expression_layout()
        stop = None
        if self.current_token[0] not in {"COLON", "COMMA", "RBRACKET"}:
            stop = self.parse_expression()
            self.skip_expression_layout()

        step = None
        has_step = False
        if self.current_token[0] == "COLON":
            has_step = True
            self.eat("COLON")
            self.skip_expression_layout()
            if self.current_token[0] not in {"COMMA", "RBRACKET"}:
                step = self.parse_expression()
                self.skip_expression_layout()

        return SliceNode(start, stop, step, has_step=has_step)

    def is_generic_constructor_suffix(self, node):
        if not isinstance(node, VariableNode) or self.current_token[0] != "LBRACKET":
            return False

        index = self.pos + 1
        depth = 1
        first_arg = None
        has_generic_marker = False

        while index < len(self.tokens):
            token_type, token_value = self.tokens[index]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    next_token = (
                        self.tokens[index + 1]
                        if index + 1 < len(self.tokens)
                        else ("EOF", None)
                    )
                    return next_token[0] == "LPAREN" and (
                        has_generic_marker or self.is_type_argument_token(first_arg)
                    )
            elif depth == 1:
                if first_arg is None and token_type not in {"COMMA"}:
                    first_arg = (token_type, token_value)
                if token_type in {"COMMA", "DOT"}:
                    has_generic_marker = True
                elif self.is_type_argument_token((token_type, token_value)):
                    has_generic_marker = True

            index += 1

        return False

    def is_type_argument_token(self, token):
        if token is None:
            return False
        token_type, token_value = token
        if token_type in {"INT", "FLOAT", "BOOL", "STRING"}:
            return True
        if token_type == "IDENTIFIER" and token_value:
            return token_value[0].isupper()
        return False

    def parse_type(self):
        if self.current_token[0] in self.FUNCTION_TYPE_TOKENS:
            return self.parse_function_type()

        if self.current_token[0] not in self.TYPE_START_TOKENS:
            raise SyntaxError(f"Expected type name, got {self.current_token[0]}")

        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        while self.current_token[0] in {"DOT", "LBRACKET", "LPAREN"}:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                if not self.is_identifier_like_token():
                    raise SyntaxError(
                        f"Expected type name after dot, got {self.current_token[0]}"
                    )
                type_name += f".{self.current_token[1]}"
                self.eat(self.current_token[0])
            elif self.current_token[0] == "LBRACKET":
                type_name += self.parse_generic_type_suffix()
            else:
                type_name += self.parse_parenthesized_type_suffix()
        if type_name.split("[", 1)[0] in self.REFERENCE_TYPE_PREFIXES:
            self.skip_layout_tokens()
            if (
                self.current_token[0]
                in self.TYPE_START_TOKENS | self.FUNCTION_TYPE_TOKENS
            ):
                type_name += f" {self.parse_type()}"
        return type_name

    def parse_function_type(self):
        type_name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACKET":
            type_name += self.parse_generic_type_suffix()
        self.skip_layout_tokens()

        if self.current_token[0] == "LPAREN":
            type_name += self.parse_parenthesized_type_suffix()
        self.skip_layout_tokens()

        while (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in self.FUNCTION_EFFECT_IDENTIFIERS
        ):
            type_name += f" {self.current_token[1]}"
            self.eat("IDENTIFIER")
            self.skip_layout_tokens()
            if self.current_token[0] == "LBRACKET":
                type_name += self.parse_generic_type_suffix()
            elif self.current_token[0] == "LPAREN":
                type_name += self.parse_parenthesized_type_suffix()
            elif self.current_token[0] == "LBRACE":
                type_name += self.parse_braced_type_suffix()
            self.skip_layout_tokens()

        if self.current_token[0] == "MINUS" and self.peek_token()[0] == "GREATER_THAN":
            self.eat("MINUS")
            self.eat("GREATER_THAN")
            type_name += f" -> {self.parse_type()}"

        return type_name

    def skip_bracketed_suffix(self):
        self.eat("LBRACKET")
        depth = 1
        while depth > 0:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated bracketed suffix")
            if self.current_token[0] == "LBRACKET":
                depth += 1
            elif self.current_token[0] == "RBRACKET":
                depth -= 1
            self.eat(self.current_token[0])

    def parse_generic_type_suffix(self):
        suffix = "["
        depth = 1
        self.eat("LBRACKET")

        while depth > 0:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated generic type argument list")

            token_type, token_value = self.current_token
            if token_type == "LBRACKET":
                suffix += "["
                depth += 1
                self.eat("LBRACKET")
            elif token_type == "RBRACKET":
                suffix += "]"
                depth -= 1
                self.eat("RBRACKET")
            elif token_type == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] != "RBRACKET":
                    suffix += ", "
            elif token_type in {"NEWLINE", "INDENT", "DEDENT"}:
                self.eat(token_type)
            else:
                suffix += token_value
                self.eat(token_type)

        return suffix

    def parse_parenthesized_type_suffix(self):
        suffix = "("
        depth = 1
        self.eat("LPAREN")

        while depth > 0:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated parenthesized type suffix")

            token_type, token_value = self.current_token
            if token_type == "LPAREN":
                suffix += "("
                depth += 1
                self.eat("LPAREN")
            elif token_type == "RPAREN":
                suffix += ")"
                depth -= 1
                self.eat("RPAREN")
            elif token_type == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] != "RPAREN":
                    suffix += ", "
            elif token_type in {"NEWLINE", "INDENT", "DEDENT"}:
                self.eat(token_type)
            else:
                suffix += token_value
                self.eat(token_type)

        return suffix

    def parse_braced_type_suffix(self):
        suffix = "{"
        depth = 1
        self.eat("LBRACE")

        while depth > 0:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated braced type suffix")

            token_type, token_value = self.current_token
            if token_type == "LBRACE":
                suffix += "{"
                depth += 1
                self.eat("LBRACE")
            elif token_type == "RBRACE":
                suffix += "}"
                depth -= 1
                self.eat("RBRACE")
            elif token_type == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] != "RBRACE":
                    suffix += ", "
            elif token_type in {"NEWLINE", "INDENT", "DEDENT"}:
                self.eat(token_type)
            else:
                suffix += token_value
                self.eat(token_type)

        return suffix

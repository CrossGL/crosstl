"""Parser for Mojo source AST construction."""

import re

from .MojoAst import *
from .MojoLexer import *


class MojoParser:
    """Parse Mojo tokens into the Mojo backend AST."""

    STATEMENT_END_TOKENS = {"NEWLINE", "DEDENT", "EOF", "RBRACE"}
    ATTRIBUTE_START_TOKENS = {"ATTRIBUTE", "AT"}
    FUNCTION_TOKENS = {"FN", "DEF"}
    ASYNC_FUNCTION_MODIFIER = "async"
    AWAIT_EXPRESSION_IDENTIFIER = "await"
    IDENTIFIER_NAME_TOKENS = {
        "IDENTIFIER",
        "DEFAULT",
        "CONSTANT",
        "BACKTICK_IDENTIFIER",
        "SWITCH",
    }
    BINDING_NAME_TOKENS = IDENTIFIER_NAME_TOKENS | {"BOOL_LITERAL"}
    TYPE_START_TOKENS = {"IDENTIFIER", "INT", "FLOAT", "BOOL", "STRING"}
    FUNCTION_TYPE_TOKENS = {"FN", "DEF"}
    ASSIGNMENT_OPERATOR_TOKENS = {
        "EQUALS",
        "PLUS_EQUALS",
        "MINUS_EQUALS",
        "MULTIPLY_EQUALS",
        "POWER_EQUALS",
        "DIVIDE_EQUALS",
        "FLOOR_DIVIDE_EQUALS",
        "ASSIGN_XOR",
        "ASSIGN_OR",
        "ASSIGN_AND",
        "ASSIGN_SHIFT_LEFT",
        "ASSIGN_SHIFT_RIGHT",
        "ASSIGN_MOD",
        "AT_EQUALS",
        "WALRUS",
    }
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
    FUNCTION_EFFECT_IDENTIFIERS = {"abi", "capturing", "raises", "thin", "unified"}
    BASE_CLASS_SPACED_TOKENS = {
        "AND",
        "OR",
        "BITWISE_AND",
        "BITWISE_OR",
        "BITWISE_XOR",
        "EQUAL",
        "NOT_EQUAL",
        "LESS_THAN",
        "GREATER_THAN",
        "LESS_EQUAL",
        "GREATER_EQUAL",
        "PLUS",
        "MINUS",
        "MULTIPLY",
        "DIVIDE",
        "FLOOR_DIVIDE",
        "MOD",
    }
    STRING_LITERAL_PREFIXES = {
        "r",
        "R",
        "t",
        "T",
        "rt",
        "rT",
        "Rt",
        "RT",
        "tr",
        "tR",
        "Tr",
        "TR",
    }

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.expression_layout_depth = 0
        self.pending_expression_layout_dedents = 0
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def skip_newlines(self):
        while self.current_token[0] == "NEWLINE":
            self.eat("NEWLINE")

    def skip_layout_tokens(self):
        while self.current_token[0] in ["NEWLINE", "INDENT", "DEDENT"]:
            if self.expression_layout_depth and self.current_token[0] == "INDENT":
                self.pending_expression_layout_dedents += 1
            elif (
                self.expression_layout_depth
                and self.current_token[0] == "DEDENT"
                and self.pending_expression_layout_dedents
            ):
                self.pending_expression_layout_dedents -= 1
            self.eat(self.current_token[0])

    def skip_expression_layout(self):
        if self.expression_layout_depth:
            self.skip_layout_tokens()

    def parse_expression_continuation(self, parse_operand):
        self.skip_expression_layout()
        return parse_operand()

    def consume_statement_terminator(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return

        if self.current_token[0] in self.STATEMENT_END_TOKENS:
            self.consume_pending_expression_layout_dedents()
            return

        raise SyntaxError(f"Expected end of statement, got {self.current_token[0]}")

    def consume_pending_expression_layout_dedents(self):
        while self.pending_expression_layout_dedents:
            if self.current_token[0] == "NEWLINE":
                self.eat("NEWLINE")
                continue
            if self.current_token[0] != "DEDENT":
                return
            if self.peek_token()[0] in {"ELIF", "ELSE"}:
                self.pending_expression_layout_dedents = 0
                return
            self.pending_expression_layout_dedents -= 1
            self.eat("DEDENT")

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

    def peek_non_layout_token(self, offset=1):
        index = self.pos + offset
        while index < len(self.tokens):
            token = self.tokens[index]
            if token[0] not in {"NEWLINE", "INDENT", "DEDENT"}:
                return token
            index += 1
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
        extensions = []
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
            elif self.is_extension_declaration_start():
                node = self.parse_extension(attributes)
                extensions.append(node)
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
            elif self.is_function_declaration_start():
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
            extensions=extensions,
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

        if self.is_dotted_name_token():
            name += self.current_token[1]
            self.eat(self.current_token[0])
        elif not name:
            raise SyntaxError(f"Expected import path, got {self.current_token[0]}")

        while self.current_token[0] == "DOT":
            self.eat("DOT")
            if not self.is_dotted_name_token():
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

    def is_identifier_name_token(self):
        return self.current_token[0] in self.IDENTIFIER_NAME_TOKENS

    def parse_identifier_name(self, context="IDENTIFIER"):
        if not self.is_identifier_name_token():
            raise SyntaxError(f"Expected {context}, got {self.current_token[0]}")

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def parse_declaration_name(self, context="declaration name"):
        if self.current_token[0] in self.TYPE_START_TOKENS:
            name = self.current_token[1]
            self.eat(self.current_token[0])
            return name
        return self.parse_identifier_name(context)

    def is_binding_name_token(self):
        return self.current_token[0] in self.BINDING_NAME_TOKENS

    def parse_binding_name(self, context="IDENTIFIER"):
        if not self.is_binding_name_token():
            raise SyntaxError(f"Expected {context}, got {self.current_token[0]}")

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def is_dotted_name_token(self):
        return self.current_token[0] == "BACKTICK_IDENTIFIER" or (
            self.is_identifier_like_token()
        )

    def parse_dotted_name(self, context):
        if not self.is_dotted_name_token():
            raise SyntaxError(f"Expected {context}, got {self.current_token[0]}")

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

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
        name = self.parse_declaration_name("struct name")
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
            base_classes.append(self.parse_base_class_entry())
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

    def parse_base_class_entry(self):
        if not self.is_base_class_entry_start():
            raise SyntaxError(f"Expected base class, got {self.current_token[0]}")

        tokens = []
        depth = 0

        while True:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated base class list")

            token_type, token_value = self.current_token
            if depth == 0 and token_type in {"COMMA", "RPAREN"}:
                break

            if token_type in {"NEWLINE", "INDENT", "DEDENT"}:
                self.eat(token_type)
                continue

            if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "RBRACE"}:
                if depth == 0:
                    break
                depth -= 1

            tokens.append((token_type, token_value))
            self.eat(token_type)

        if not tokens:
            raise SyntaxError("Expected base class or trait conformance")
        return self.format_base_class_entry(tokens)

    def is_base_class_entry_start(self):
        return self.current_token[0] in self.FUNCTION_TYPE_TOKENS or (
            self.is_base_class_word_token(self.current_token[0])
        )

    def format_base_class_entry(self, tokens):
        expression = ""
        previous_type = None
        previous_value = None

        for token_type, token_value in tokens:
            value = str(token_value)
            if self.needs_base_class_token_space(
                previous_type, previous_value, token_type, value
            ):
                expression += " "
            expression += value
            previous_type = token_type
            previous_value = value

        return expression

    def needs_base_class_token_space(
        self, previous_type, previous_value, token_type, token_value
    ):
        if previous_type is None:
            return False
        if token_type in {"COMMA", "RPAREN", "RBRACKET", "RBRACE", "DOT", "COLON"}:
            return False
        if previous_type in {"LPAREN", "LBRACKET", "LBRACE", "DOT"}:
            return False
        if previous_type == "COMMA":
            return True
        if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
            return False
        if previous_value == "where" or token_value == "where":
            return True
        if token_type in self.BASE_CLASS_SPACED_TOKENS:
            return True
        if previous_type in self.BASE_CLASS_SPACED_TOKENS:
            return True
        return self.is_base_class_word_token(
            previous_type
        ) and self.is_base_class_word_token(token_type)

    def is_base_class_word_token(self, token_type):
        return token_type in self.TYPE_START_TOKENS | {
            "IDENTIFIER",
            "DEFAULT",
            "BACKTICK_IDENTIFIER",
            "BOOL_LITERAL",
            "NUMBER",
            "STRING_LITERAL",
        }

    def parse_struct_member(self):
        self.skip_newlines()
        attributes = self.parse_attributes()
        if self.is_ellipsis_statement():
            return self.parse_ellipsis_statement()
        if self.current_token[0] == "STRING_LITERAL":
            return self.parse_standalone_string_literal_statement()
        if self.current_token[0] == "PASS":
            self.eat("PASS")
            self.consume_statement_terminator()
            return PassNode()
        if self.current_token[0] == "STRUCT":
            return self.parse_struct(attributes)
        if self.current_token[0] == "CLASS":
            return self.parse_class(attributes)
        if self.current_token[0] == "TRAIT":
            return self.parse_trait(attributes)
        if self.is_function_declaration_start():
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
        name = self.parse_declaration_name("class name")
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
        name = self.parse_declaration_name("trait name")
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

    def is_extension_declaration_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "__extension"
        )

    def parse_extension(self, initial_attributes=None):
        self.eat("IDENTIFIER")
        name = self.parse_type()
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
            raise SyntaxError(f"Expected extension body, got {self.current_token[0]}")

        return ExtensionNode(
            name,
            members=members,
            methods=methods,
            attributes=initial_attributes,
        )

    def parse_class_member(self):
        self.skip_newlines()
        attributes = self.parse_attributes()

        if self.is_ellipsis_statement():
            return self.parse_ellipsis_statement()
        if self.current_token[0] == "STRING_LITERAL":
            return self.parse_standalone_string_literal_statement()
        if self.current_token[0] == "PASS":
            self.eat("PASS")
            self.consume_statement_terminator()
            return PassNode()
        if self.is_function_declaration_start():
            return self.parse_function(attributes)
        if self.current_token[0] == "STRUCT":
            return self.parse_struct(attributes)
        if self.current_token[0] == "CLASS":
            return self.parse_class(attributes)
        if self.current_token[0] == "TRAIT":
            return self.parse_trait(attributes)
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
        is_async = False
        if self.is_async_function_modifier():
            is_async = True
            self.eat("IDENTIFIER")
            self.skip_layout_tokens()

        if self.current_token[0] in self.FUNCTION_TOKENS:
            self.eat(self.current_token[0])
        else:
            raise SyntaxError(
                f"Expected function declaration, got {self.current_token[0]}"
            )

        return_type = None
        if self.current_token[0] not in self.IDENTIFIER_NAME_TOKENS:
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
        func.is_async = is_async
        return func

    def is_function_declaration_start(self):
        if self.current_token[0] in self.FUNCTION_TOKENS:
            return True
        return self.is_async_function_modifier()

    def is_async_function_modifier(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == self.ASYNC_FUNCTION_MODIFIER
            and self.peek_token()[0] in self.FUNCTION_TOKENS
        )

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
            effect_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.skip_layout_tokens()
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_group("LPAREN", "RPAREN")
            elif self.current_token[0] == "LBRACE":
                self.skip_balanced_group("LBRACE", "RBRACE")
            elif effect_name == "raises":
                self.parse_optional_raises_error_type()
            self.skip_layout_tokens()

    def parse_optional_raises_error_type(self):
        if not self.is_raises_error_type_start():
            return None
        return self.parse_type()

    def is_raises_error_type_start(self):
        if self.current_token[0] in self.STATEMENT_END_TOKENS | {"COLON", "MINUS"}:
            return False
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] in (
            self.FUNCTION_EFFECT_IDENTIFIERS | {"where"}
        ):
            return False
        return self.current_token[0] in (
            self.TYPE_START_TOKENS | self.FUNCTION_TYPE_TOKENS
        )

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

        return self.parse_unparenthesized_where_clause()

    def parse_unparenthesized_where_clause(self):
        parts = []
        depth = 0

        while True:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated Mojo where clause")

            token_type, token_value = self.current_token
            if token_type in {"NEWLINE", "INDENT", "DEDENT"}:
                self.eat(token_type)
                continue

            if token_type == "COLON" and depth == 0:
                break

            if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "RBRACE"}:
                if depth == 0:
                    break
                depth -= 1

            parts.append(str(token_value))
            self.eat(token_type)

        if not parts:
            raise SyntaxError("Expected Mojo where clause constraint")
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
            variadic_kind = self.parse_variadic_parameter_marker()

            if self.is_untyped_self_parameter(convention):
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                vtype = ""
            elif self.is_bare_self_parameter():
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                vtype = ""
            elif self.is_identifier_name_token() or (
                self.current_token[0] in self.TYPE_START_TOKENS
            ):
                if self.is_identifier_name_token() and self.peek_token()[0] == "COLON":
                    name = self.current_token[1]
                    self.eat(self.current_token[0])
                    self.eat("COLON")
                    vtype = self.parse_type()
                elif self.current_token[0] in self.TYPE_START_TOKENS:
                    vtype = self.parse_type()
                    name = ""
                    if self.current_token[0] == "IDENTIFIER":
                        name = self.current_token[1]
                        self.eat("IDENTIFIER")
                else:
                    raise SyntaxError(
                        f"Unexpected token in parameter list: {self.current_token[0]}"
                    )

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
            param.is_variadic = variadic_kind is not None
            param.is_variadic_keyword = variadic_kind == "keyword"
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

        next_token = self.peek_non_layout_token()[0]
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
        if next_token[0] in self.TYPE_START_TOKENS | self.IDENTIFIER_NAME_TOKENS:
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
        if self.current_token[0] == "POWER":
            if self.peek_token()[0] not in self.TYPE_START_TOKENS:
                return None
            self.eat("POWER")
            self.skip_layout_tokens()
            return "keyword"
        if self.current_token[0] != "MULTIPLY":
            return None
        if self.peek_token()[0] not in self.TYPE_START_TOKENS:
            return None
        self.eat("MULTIPLY")
        self.skip_layout_tokens()
        return "positional"

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
        self.consume_block_header_continuation_dedents()
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
                if self.is_block_clause_terminator():
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

    def consume_block_header_continuation_dedents(self):
        while self.current_token[0] == "DEDENT" and self.has_pending_body_indent():
            self.eat("DEDENT")
            self.skip_newlines()

    def has_pending_body_indent(self):
        index = self.pos
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "NEWLINE":
                index += 1
                continue
            if token_type == "DEDENT":
                index += 1
                continue
            return token_type == "INDENT"
        return False

    def is_block_clause_terminator(self):
        if self.current_token[0] in {"ELSE", "ELIF"}:
            return True
        return self.current_token[0] == "IDENTIFIER" and self.current_token[1] in {
            "except",
            "finally",
        }

    def parse_statement(self):
        self.skip_newlines()
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "try":
            return self.parse_try_except_statement()
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "assert":
            return self.parse_assert_statement()
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "__mlir_region"
        ):
            return self.parse_mlir_region_statement()
        if self.current_token[0] in ["IMPORT", "FROM"]:
            return self.parse_import_statement()
        if self.is_function_declaration_start():
            return self.parse_function()

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
            "LPAREN",
        ]:
            return self.parse_variable_declaration_or_assignment()

        elif self.current_token[0] in self.ATTRIBUTE_START_TOKENS:
            attributes = self.parse_attributes()
            if self.is_function_declaration_start():
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

        elif self.is_function_declaration_start():
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
        elif self.current_token[0] == "STRING_LITERAL":
            return self.parse_standalone_string_literal_statement()
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

    def parse_standalone_string_literal_statement(self):
        self.eat("STRING_LITERAL")
        self.consume_statement_terminator()
        return PassNode()

    def parse_variable_declaration_or_assignment(self):
        if self.current_token[0] in ["COMPTIME", "ALIAS"]:
            return self.parse_comptime_or_alias_statement()

        if self.is_ref_binding_declaration_start():
            return self.parse_ref_binding_declaration()

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
        if self.current_token[0] == "COLON" and self.is_typed_assignment_target(
            statement
        ):
            statement = self.parse_typed_assignment_tail(statement)
        if self.current_token[0] == "COMMA":
            statement = self.parse_tuple_assignment_tail(statement)
        self.consume_statement_terminator()
        return statement

    def is_typed_assignment_target(self, target):
        return isinstance(target, (ArrayAccessNode, MemberAccessNode, VariableNode))

    def parse_typed_assignment_tail(self, target):
        self.eat("COLON")
        target_type = self.parse_type()
        if self.current_token[0] != "EQUALS":
            raise SyntaxError("Expected assignment after typed assignment target")
        self.eat("EQUALS")
        value = self.parse_expression_list_value()
        assignment = AssignmentNode(target, value, "=")
        assignment.target_type = target_type
        return assignment

    def parse_tuple_assignment_tail(self, first_target):
        targets = [first_target]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            self.skip_layout_tokens()
            targets.append(self.parse_tuple_assignment_target())
            self.skip_expression_layout()

        if self.current_token[0] not in self.ASSIGNMENT_OPERATOR_TOKENS:
            raise SyntaxError("Expected assignment after expression tuple")

        op = self.current_token[1]
        self.eat(self.current_token[0])
        right = self.parse_expression_list_value()
        return AssignmentNode(TupleNode(targets), right, op)

    def parse_tuple_assignment_target(self):
        convention = self.parse_parameter_convention()
        target = self.parse_logical_or()
        if convention:
            target.target_convention = convention
        return target

    def parse_binding_convention_target_expression(self):
        convention = self.parse_parameter_convention()
        if not convention:
            raise SyntaxError(
                f"Expected binding convention, got {self.current_token[0]}"
            )

        name = self.parse_binding_name("binding target")
        target = VariableNode("", name)
        target.target_convention = convention
        return target

    def is_ref_binding_declaration_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "ref"
            and self.peek_token()[0] in self.BINDING_NAME_TOKENS
        )

    def parse_ref_binding_declaration(self):
        self.eat("IDENTIFIER")
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
        node = VariableDeclarationNode(
            vtype,
            name,
            initial_value,
            is_var=True,
            attributes=attributes,
        )
        node.binding_convention = "ref"
        return node

    def parse_identifier_tuple(self):
        parenthesized = self.current_token[0] == "LPAREN"
        saw_comma = False
        if parenthesized:
            self.eat("LPAREN")
            self.skip_layout_tokens()

        identifiers = [self.parse_identifier_tuple_element()]
        if parenthesized:
            self.skip_layout_tokens()

        while self.current_token[0] == "COMMA":
            saw_comma = True
            self.eat("COMMA")
            if parenthesized:
                self.skip_layout_tokens()
                if self.current_token[0] == "RPAREN":
                    break
            identifiers.append(self.parse_identifier_tuple_element())
            if parenthesized:
                self.skip_layout_tokens()

        if parenthesized:
            self.eat("RPAREN")

        if len(identifiers) == 1 and not saw_comma:
            return identifiers[0].name
        return TupleNode(identifiers)

    def parse_identifier_tuple_element(self):
        if self.current_token[0] == "LPAREN":
            return self.parse_identifier_tuple()
        return VariableNode("", self.parse_binding_name("IDENTIFIER after comma"))

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
        if self.is_comptime_tuple_declaration_start():
            return self.parse_comptime_tuple_declaration(after_keyword=True)
        if self.is_comptime_declaration_start():
            return self.parse_comptime_declaration(after_keyword=True)
        if self.is_comptime_expression_statement():
            node = self.parse_expression()
            self.consume_statement_terminator()
            node.is_comptime = True
            return node
        return self.parse_comptime_declaration(after_keyword=True)

    def is_comptime_declaration_start(self):
        if not self.is_identifier_name_token():
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

    def is_comptime_tuple_declaration_start(self):
        return self.is_identifier_name_token() and self.peek_token()[0] == "COMMA"

    def parse_comptime_tuple_declaration(self, after_keyword=False):
        if not after_keyword:
            self.eat("COMPTIME")

        name = self.parse_identifier_tuple()
        if self.current_token[0] != "EQUALS":
            raise SyntaxError("Expected assignment after comptime identifier tuple")
        self.eat("EQUALS")
        initial_value = self.parse_expression_list_value()
        self.consume_statement_terminator()
        node = VariableDeclarationNode(None, name, initial_value, is_var=False)
        node.is_comptime = True
        return node

    def parse_assert_statement(self, is_comptime=False):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN" and not self.assert_condition_continues():
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

    def assert_condition_continues(self):
        if self.current_token[0] != "LPAREN":
            return False

        index = self.pos
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    index += 1
                    if index >= len(self.tokens):
                        return False
                    return self.tokens[index][0] not in (
                        self.STATEMENT_END_TOKENS | {"COMMA", "SEMICOLON"}
                    )
            index += 1

        return False

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

        name = self.parse_identifier_name("comptime declaration name")
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

    def parse_mlir_region_statement(self):
        self.eat("IDENTIFIER")
        self.parse_identifier_name("MLIR region name")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_group("LPAREN", "RPAREN")
        self.eat("COLON")
        self.parse_block()
        return PassNode()

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
            node = ForNode(init, condition, update, body)
            node.else_body = self.parse_loop_else_body()
            return node

        except Exception:
            self.pos = saved_pos
            self.current_token = saved_token

        convention = self.parse_parameter_convention()
        if self.is_identifier_name_token():
            var_name = self.parse_identifier_tuple()
            if self.current_token[0] == "IN":
                self.eat("IN")
                iterable = self.parse_for_iterable()
                self.eat("COLON")
                body = self.parse_block()
                node = RangeForNode("", var_name, iterable, body)
                node.target_convention = convention
                node.else_body = self.parse_loop_else_body()
                return node

        raise SyntaxError(
            "Invalid for loop syntax. Expected C-style "
            "'for init; condition; update:' or Python-style "
            "'for [convention] target in iterable:'"
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
        node = WhileNode(condition, body)
        node.else_body = self.parse_loop_else_body()
        return node

    def parse_loop_else_body(self):
        self.skip_newlines()
        if self.current_token[0] != "ELSE":
            return []

        self.eat("ELSE")
        self.eat("COLON")
        return self.parse_block()

    def parse_with_statement(self):
        self.eat("WITH")
        self.skip_layout_tokens()

        contexts = []
        while True:
            context_expr = self.parse_expression()
            alias = None
            if isinstance(context_expr, CastNode) and self.current_token[0] in {
                "COLON",
                "COMMA",
            }:
                alias = context_expr.target_type
                context_expr = context_expr.expression
            elif self.current_token[0] == "AS":
                self.eat("AS")
                alias = self.current_token[1]
                self.eat("IDENTIFIER")

            contexts.append((context_expr, alias))
            self.skip_layout_tokens()
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
            self.skip_layout_tokens()

        self.eat("COLON")
        body = self.parse_block()
        context_expr, alias = contexts[0]
        return WithNode(context_expr, alias, body, contexts=contexts)

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
            value = self.parse_expression_list_value()

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
        if self.current_token[0] in self.ASSIGNMENT_OPERATOR_TOKENS:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_expression_continuation(self.parse_assignment)
            right = self.parse_indexed_call_continuation(right)
            return AssignmentNode(left, right, op)
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression_continuation(self.parse_expression)
            self.eat("COLON")
            false_expr = self.parse_expression_continuation(self.parse_expression)
            left = TernaryOpNode(left, true_expr, false_expr)
        self.skip_expression_layout()
        if self.current_token[0] == "IF" and self.is_inline_if_expression():
            true_expr = left
            self.eat("IF")
            condition = self.parse_expression_continuation(self.parse_expression)
            self.eat("ELSE")
            false_expr = self.parse_expression_continuation(self.parse_expression)
            left = TernaryOpNode(condition, true_expr, false_expr)
        return self.parse_indexed_call_continuation(left)

    def parse_indexed_call_continuation(self, node):
        if (
            isinstance(node, ArrayAccessNode)
            and self.current_token[0] == "NEWLINE"
            and self.peek_non_layout_token()[0] == "LPAREN"
        ):
            self.skip_layout_tokens()
            return self.parse_postfix_suffixes(node)
        return node

    def is_inline_if_expression(self):
        layout_tokens = {"NEWLINE", "INDENT", "DEDENT"}
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
                if token_type in layout_tokens and self.expression_layout_depth:
                    idx += 1
                    continue
                if token_type in layout_tokens | {"EOF", "COMMA", "SEMICOLON"}:
                    return False
            idx += 1
        return False

    def parse_logical_or(self):
        left = self.parse_logical_and()
        self.skip_expression_layout()
        while self.current_token[0] == "OR":
            op = self.current_token[1]
            self.eat("OR")
            right = self.parse_expression_continuation(self.parse_logical_and)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_logical_and(self):
        left = self.parse_bitwise_or()
        self.skip_expression_layout()
        while self.current_token[0] == "AND":
            op = self.current_token[1]
            self.eat("AND")
            right = self.parse_expression_continuation(self.parse_bitwise_or)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_bitwise_or(self):
        left = self.parse_bitwise_xor()
        self.skip_expression_layout()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_expression_continuation(self.parse_bitwise_xor)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_bitwise_xor(self):
        left = self.parse_bitwise_and()
        self.skip_expression_layout()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_expression_continuation(self.parse_bitwise_and)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_bitwise_and(self):
        left = self.parse_equality()
        self.skip_expression_layout()
        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_expression_continuation(self.parse_equality)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_equality(self):
        left = self.parse_relational()
        self.skip_expression_layout()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_expression_continuation(self.parse_relational)
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
            or self.is_negated_membership_operator()
            or self.is_identity_operator()
        ):
            if self.is_negated_membership_operator():
                op = self.parse_negated_membership_operator()
            elif self.is_identity_operator():
                op = self.parse_identity_operator()
            else:
                op = self.current_token[1]
                self.eat(self.current_token[0])
            right = self.parse_expression_continuation(self.parse_shift)
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

    def is_negated_membership_operator(self):
        return self.current_token[0] == "NOT" and self.peek_token()[0] == "IN"

    def parse_negated_membership_operator(self):
        self.eat("NOT")
        self.eat("IN")
        return "not in"

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
            right = self.parse_expression_continuation(self.parse_additive)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        self.skip_expression_layout()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_expression_continuation(self.parse_multiplicative)
            self.skip_expression_layout()
            right = self.parse_adjacent_string_literals(right)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        self.skip_expression_layout()
        while self.current_token[0] in [
            "MULTIPLY",
            "AT",
            "DIVIDE",
            "FLOOR_DIVIDE",
            "MOD",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_expression_continuation(self.parse_unary)
            left = BinaryOpNode(left, op, right)
            self.skip_expression_layout()
        return left

    def parse_unary(self):
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == self.AWAIT_EXPRESSION_IDENTIFIER
        ):
            self.eat("IDENTIFIER")
            operand = self.parse_unary()
            return UnaryOpNode("await", operand)
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
        if self.current_token[0] == "POWER":
            self.eat("POWER")
            operand = self.parse_unary()
            return SpreadExpressionNode(operand, kind="keyword")
        if self.current_token[0] == "MULTIPLY":
            self.eat("MULTIPLY")
            operand = self.parse_unary()
            return SpreadExpressionNode(operand)
        return self.parse_power()

    def parse_power(self):
        left = self.parse_primary()
        if self.current_token[0] == "POWER":
            op = self.current_token[1]
            self.eat("POWER")
            right = self.parse_expression_continuation(self.parse_unary)
            return BinaryOpNode(left, op, right)
        return left

    def parse_grouped_expression_item(self):
        self.expression_layout_depth += 1
        try:
            expr = self.parse_expression()
            self.skip_layout_tokens()
            return self.parse_adjacent_string_literals(expr)
        finally:
            self.expression_layout_depth -= 1

    def parse_primary(self):
        if self.is_binding_convention_target_expression_start():
            return self.parse_binding_convention_target_expression()
        if self.current_token[0] in self.IDENTIFIER_NAME_TOKENS:
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return self.parse_postfix_suffixes(value)
        elif self.current_token[0] == "STRING_LITERAL":
            value = self.current_token[1]
            self.eat("STRING_LITERAL")
            self.skip_expression_layout()
            value = self.parse_adjacent_string_literals(value)
            return self.parse_postfix_suffixes(value)
        elif self.current_token[0] == "BOOL_LITERAL":
            value = self.current_token[1]
            self.eat("BOOL_LITERAL")
            return self.parse_postfix_suffixes(value)
        elif self.is_ellipsis_expression():
            return self.parse_ellipsis_expression()
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
            if self.current_token[0] == "RPAREN":
                self.eat("RPAREN")
                return self.parse_postfix_suffixes(TupleNode([]))
            expr = self.parse_grouped_expression_item()
            self.skip_layout_tokens()
            if self.current_token[0] == "COMMA":
                elements = [expr]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    self.skip_layout_tokens()
                    if self.current_token[0] == "RPAREN":
                        break
                    elements.append(self.parse_grouped_expression_item())
                    self.skip_layout_tokens()
                self.eat("RPAREN")
                return self.parse_postfix_suffixes(TupleNode(elements))
            self.eat("RPAREN")
            return self.parse_postfix_suffixes(expr)
        elif self.current_token[0] == "LBRACKET":
            return self.parse_list_literal()
        elif self.current_token[0] == "LBRACE":
            return self.parse_dict_literal()
        elif self.current_token[0] in self.FUNCTION_TYPE_TOKENS:
            return self.parse_function_type()
        elif self.current_token[0] in ["STRUCT", "CLASS", "LET", "VAR"]:
            raise SyntaxError(f"Unexpected top-level keyword: {self.current_token[0]}")
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def is_binding_convention_target_expression_start(self):
        return self.current_token[0] == "VAR" or (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in self.PARAMETER_CONVENTION_IDENTIFIERS
            and self.is_parameter_convention_followed_by_parameter()
        )

    def is_ellipsis_expression(self):
        return (
            self.current_token[0] == "DOT"
            and self.peek_token()[0] == "DOT"
            and self.peek_token(2)[0] == "DOT"
        )

    def parse_ellipsis_expression(self):
        self.eat("DOT")
        self.eat("DOT")
        self.eat("DOT")
        return VariableNode("", "...")

    def parse_list_literal(self):
        self.eat("LBRACKET")
        elements = []
        self.skip_layout_tokens()

        while self.current_token[0] != "RBRACKET":
            self.expression_layout_depth += 1
            try:
                element = self.parse_expression()
            finally:
                self.expression_layout_depth -= 1
            elements.append(element)
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

    def parse_dict_literal(self):
        self.eat("LBRACE")
        entries = []
        elements = []
        self.skip_layout_tokens()

        if self.current_token[0] == "RBRACE":
            self.eat("RBRACE")
            return self.parse_postfix_suffixes(DictLiteralNode(entries))

        while self.current_token[0] != "RBRACE":
            self.expression_layout_depth += 1
            try:
                key = self.parse_expression()
            finally:
                self.expression_layout_depth -= 1
            self.skip_layout_tokens()

            if self.current_token[0] == "COLON":
                if elements:
                    raise SyntaxError("Cannot mix Mojo dictionary and braced elements")
                self.eat("COLON")
                self.skip_layout_tokens()
                self.expression_layout_depth += 1
                try:
                    value = self.parse_expression()
                finally:
                    self.expression_layout_depth -= 1
                self.skip_layout_tokens()

                if not entries and self.current_token[0] == "FOR":
                    comprehension = self.parse_dict_comprehension(key, value)
                    self.eat("RBRACE")
                    return self.parse_postfix_suffixes(comprehension)

                entries.append((key, value))
            else:
                if entries:
                    raise SyntaxError("Cannot mix Mojo dictionary and braced elements")
                if not elements and self.current_token[0] == "FOR":
                    comprehension = self.parse_set_comprehension(key)
                    self.eat("RBRACE")
                    return self.parse_postfix_suffixes(comprehension)
                elements.append(key)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] == "RBRACE":
                    break
            elif self.current_token[0] != "RBRACE":
                raise SyntaxError(
                    f"Expected COMMA or RBRACE, got {self.current_token[0]}"
                )

        self.eat("RBRACE")
        if entries:
            return self.parse_postfix_suffixes(DictLiteralNode(entries))
        return self.parse_postfix_suffixes(BracedLiteralNode(elements))

    def parse_list_comprehension(self, expression):
        clauses = self.parse_comprehension_clauses()
        return ListComprehensionNode(expression, clauses)

    def parse_dict_comprehension(self, key, value):
        clauses = self.parse_comprehension_clauses()
        return DictComprehensionNode(key, value, clauses)

    def parse_set_comprehension(self, expression):
        clauses = self.parse_comprehension_clauses()
        return SetComprehensionNode(expression, clauses)

    def parse_comprehension_clauses(self):
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

        return clauses

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
        self.expression_layout_depth += 1
        try:
            self.skip_layout_tokens()
            while self.current_token[0] != "RPAREN":
                arg = self.parse_expression()
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
        finally:
            self.expression_layout_depth -= 1
        return VectorConstructorNode(type_name, args)

    def parse_function_call_or_identifier(self):
        if self.current_token[0] in self.IDENTIFIER_NAME_TOKENS:
            name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[
                0
            ] == "STRING_LITERAL" and self.is_string_literal_prefix(name):
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
                member = self.parse_dotted_name("IDENTIFIER after dot")
                node = MemberAccessNode(node, member)
                continue

            if self.current_token[0] == "LBRACKET":
                if self.is_generic_constructor_suffix(node):
                    type_name = node.name + self.parse_generic_type_suffix()
                    node = self.parse_vector_constructor(type_name)
                else:
                    node = self.parse_array_access(node)
                continue

            if self.current_token[0] == "LBRACE":
                node = self.parse_braced_initializer_suffix(node)
                continue

            if self.current_token[0] == "ATTRIBUTE":
                node = ArrayAccessNode(node, self.parse_attribute_index_suffix())
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

    def parse_braced_initializer_suffix(self, callee):
        self.eat("LBRACE")
        args = []
        self.expression_layout_depth += 1
        try:
            self.skip_layout_tokens()
            while self.current_token[0] != "RBRACE":
                key = self.parse_expression()
                self.skip_layout_tokens()
                if self.current_token[0] == "COLON":
                    self.eat("COLON")
                    self.skip_layout_tokens()
                    value = self.parse_expression()
                    key = AssignmentNode(key, value, "=")
                args.append(key)
                self.skip_layout_tokens()
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    self.skip_layout_tokens()
                    if self.current_token[0] == "RBRACE":
                        break
                elif self.current_token[0] != "RBRACE":
                    raise SyntaxError(
                        f"Expected COMMA or RBRACE, got {self.current_token[0]}"
                    )
            self.eat("RBRACE")
        finally:
            self.expression_layout_depth -= 1

        if isinstance(callee, VariableNode):
            return FunctionCallNode(callee.name, args)
        return CallNode(callee, args)

    def parse_attribute_index_suffix(self):
        content = self.current_token[1][2:-2].strip()
        self.eat("ATTRIBUTE")
        if not content:
            return TupleNode([])

        parser = MojoParser(MojoLexer(f"[{content}]").tokenize())
        index = parser.parse_array_access(VariableNode("", "__attribute__")).index
        parser.skip_layout_tokens()
        parser.eat("EOF")
        return index

    def is_postfix_transfer_marker(self):
        return self.peek_token()[0] in self.STATEMENT_END_TOKENS | {
            "COMMA",
            "DOT",
            "RPAREN",
            "RBRACKET",
            "COLON",
        }

    def parse_call(self, callee):
        self.eat("LPAREN")
        args = []
        self.expression_layout_depth += 1
        try:
            self.skip_layout_tokens()
            while self.current_token[0] != "RPAREN":
                arg = self.parse_expression()
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
        finally:
            self.expression_layout_depth -= 1
        if isinstance(callee, VariableNode):
            return FunctionCallNode(callee.name, args)
        if isinstance(callee, MemberAccessNode):
            return MethodCallNode(callee.object, callee.member, args)
        return CallNode(callee, args)

    def parse_adjacent_string_literals(self, arg):
        if not self.is_string_literal_value(arg):
            return arg

        while True:
            self.consume_adjacent_string_continuation_layout()
            if not self.is_adjacent_string_literal_start():
                break
            next_literal = self.parse_adjacent_string_literal()
            arg = self.concat_string_literals(arg, next_literal)
        return arg

    def consume_adjacent_string_continuation_layout(self):
        if self.expression_layout_depth and self.current_token[0] == "NEWLINE":
            index = self.pos + 1
            while index < len(self.tokens) and self.tokens[index][0] in {
                "NEWLINE",
                "INDENT",
                "DEDENT",
            }:
                index += 1
            if index < len(self.tokens) and self.is_adjacent_string_token_at(index):
                self.skip_layout_tokens()
            return

        if self.current_token[0] != "NEWLINE" or self.peek_token()[0] != "INDENT":
            return

        index = self.pos + 2
        while index < len(self.tokens) and self.tokens[index][0] == "NEWLINE":
            index += 1
        if index >= len(self.tokens):
            return

        if not self.is_adjacent_string_token_at(index):
            return

        self.eat("NEWLINE")
        self.eat("INDENT")
        self.pending_expression_layout_dedents += 1
        while self.current_token[0] == "NEWLINE":
            self.eat("NEWLINE")

    def is_adjacent_string_token_at(self, index):
        token_type, token_value = self.tokens[index]
        return token_type == "STRING_LITERAL" or (
            token_type in self.IDENTIFIER_NAME_TOKENS
            and self.is_string_literal_prefix(token_value)
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1][0] == "STRING_LITERAL"
        )

    def is_adjacent_string_literal_start(self):
        if self.current_token[0] == "STRING_LITERAL":
            return True
        return (
            self.current_token[0] in self.IDENTIFIER_NAME_TOKENS
            and self.is_string_literal_prefix(self.current_token[1])
            and self.peek_token()[0] == "STRING_LITERAL"
        )

    def parse_adjacent_string_literal(self):
        prefix = ""
        if self.current_token[0] in self.IDENTIFIER_NAME_TOKENS:
            prefix = self.current_token[1]
            self.eat(self.current_token[0])
        literal = self.current_token[1]
        self.eat("STRING_LITERAL")
        return f"{prefix}{literal}"

    def is_string_literal_prefix(self, value):
        return isinstance(value, str) and value in self.STRING_LITERAL_PREFIXES

    def is_string_literal_value(self, value):
        return self.split_string_literal(value) is not None

    def concat_string_literals(self, left, right):
        left_parts = self.split_string_literal(left)
        right_parts = self.split_string_literal(right)
        if left_parts is None or right_parts is None:
            return left

        left_prefix, left_quote, left_body = left_parts
        _, _, right_body = right_parts
        return f"{left_prefix}{left_quote}{left_body}{right_body}{left_quote}"

    def split_string_literal(self, value):
        if not isinstance(value, str):
            return None

        for quote in ("'", '"'):
            quote_index = value.find(quote)
            if quote_index == -1:
                continue
            prefix = value[:quote_index]
            if prefix and not self.is_string_literal_prefix(prefix):
                continue
            if not value.endswith(quote):
                continue
            return prefix, quote, value[quote_index + 1 : -1]
        return None

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
        self.expression_layout_depth += 1
        try:
            self.skip_layout_tokens()
            if self.current_token[0] == "RBRACKET":
                self.eat("RBRACKET")
                return ArrayAccessNode(array, TupleNode([]))

            indices = [self.parse_index_component()]
            self.skip_layout_tokens()
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] == "RBRACKET":
                    break
                indices.append(self.parse_index_component())
                self.skip_layout_tokens()
            self.eat("RBRACKET")
        finally:
            self.expression_layout_depth -= 1
        index = indices[0] if len(indices) == 1 else TupleNode(indices)
        return ArrayAccessNode(array, index)

    def parse_index_component(self):
        if (
            self.current_token[0] in self.IDENTIFIER_NAME_TOKENS
            and self.peek_token()[0] == "EQUALS"
        ):
            keyword = VariableNode("", self.parse_identifier_name("index keyword"))
            self.eat("EQUALS")
            return AssignmentNode(keyword, self.parse_index_component())

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
        if (
            self.current_token[0] == "MULTIPLY"
            and self.peek_token()[0] in self.TYPE_START_TOKENS
        ):
            self.eat("MULTIPLY")
            return f"*{self.parse_type()}"

        if self.current_token[0] in self.FUNCTION_TYPE_TOKENS:
            return self.parse_function_type()

        if self.current_token[0] == "LPAREN":
            return self.parse_parenthesized_type_group()

        if self.current_token[0] not in self.TYPE_START_TOKENS:
            raise SyntaxError(f"Expected type name, got {self.current_token[0]}")

        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        while self.current_token[0] in {"DOT", "LBRACKET", "LPAREN"}:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                type_name += f".{self.parse_dotted_name('type name after dot')}"
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
        while self.current_token[0] == "BITWISE_AND":
            self.eat("BITWISE_AND")
            self.skip_layout_tokens()
            type_name += f"&{self.parse_type()}"
        return type_name

    def parse_parenthesized_type_group(self):
        self.eat("LPAREN")
        self.skip_layout_tokens()

        types = []
        while self.current_token[0] != "RPAREN":
            types.append(self.parse_type())
            self.skip_layout_tokens()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_layout_tokens()
                if self.current_token[0] == "RPAREN":
                    break
                continue
            break

        self.eat("RPAREN")
        if len(types) == 1:
            return types[0]
        return f"Tuple[{', '.join(types)}]"

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
            effect_name = self.current_token[1]
            type_name += f" {effect_name}"
            self.eat("IDENTIFIER")
            if self.is_function_type_effect_boundary():
                return type_name
            self.skip_layout_tokens()
            if self.current_token[0] == "LBRACKET":
                type_name += self.parse_generic_type_suffix()
            elif self.current_token[0] == "LPAREN":
                type_name += self.parse_parenthesized_type_suffix()
            elif self.current_token[0] == "LBRACE":
                type_name += self.parse_braced_type_suffix()
            elif effect_name == "raises":
                error_type = self.parse_optional_raises_error_type()
                if error_type:
                    type_name += f" {error_type}"
            self.skip_layout_tokens()

        if self.current_token[0] == "MINUS" and self.peek_token()[0] == "GREATER_THAN":
            self.eat("MINUS")
            self.eat("GREATER_THAN")
            type_name += f" -> {self.parse_type()}"

        return type_name

    def is_function_type_effect_boundary(self):
        token_type = self.current_token[0]
        if token_type in {"COMMA", "RPAREN", "RBRACKET", "SEMICOLON"}:
            return True
        if token_type in self.STATEMENT_END_TOKENS:
            next_type = self.peek_non_layout_token()[0]
            return next_type not in {"MINUS", "LBRACKET", "LPAREN", "LBRACE"}
        return False

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
            elif token_type == "INDENT":
                self.eat("INDENT")
            elif token_type == "DEDENT":
                self.eat("DEDENT")
            elif token_type in {"IF", "ELSE"}:
                suffix = suffix.rstrip() + f" {token_value} "
                self.eat(token_type)
            elif token_type == "NEWLINE":
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
            elif token_type == "INDENT":
                self.eat("INDENT")
            elif token_type == "DEDENT":
                self.eat("DEDENT")
            elif token_type == "NEWLINE":
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
            elif token_type == "INDENT":
                self.eat("INDENT")
            elif token_type == "DEDENT":
                self.eat("DEDENT")
            elif token_type == "NEWLINE":
                self.eat(token_type)
            else:
                suffix += token_value
                self.eat(token_type)

        return suffix

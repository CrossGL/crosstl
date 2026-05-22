"""Parser for Vulkan SPIR-V source AST construction."""

from .VulkanLexer import *
from .VulkanAst import *


class VulkanParser:
    """Parse Vulkan/SPIR-V style tokens into the Vulkan backend AST."""

    PARAMETER_QUALIFIER_TOKENS = {"IN", "OUT", "INOUT"}
    PRECISION_QUALIFIER_TOKENS = {"HIGHP", "MEDIUMP", "LOWP"}
    LAYOUT_DECLARATION_QUALIFIERS = {
        "centroid",
        "coherent",
        "flat",
        "highp",
        "invariant",
        "lowp",
        "mediump",
        "noperspective",
        "patch",
        "pervertexEXT",
        "precise",
        "readonly",
        "restrict",
        "sample",
        "smooth",
        "volatile",
        "writeonly",
    }

    def __init__(self, tokens):
        """Initialize the parser with a token stream from ``VulkanLexer``."""
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.loop_depth = 0
        self.breakable_depth = 0
        self.skip_comments()

    def skip_comments(self):
        """Advance past comment tokens before parsing syntax."""
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def peek(self, offset):
        """Look ahead by offset tokens without consuming them."""
        peek_index = self.pos + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index][
                0
            ]  # Return the type of the token at the peeked index
        return None

    def skip_until(self, token_type):
        """Skip tokens until the specified token type is found"""
        while self.current_token[0] != token_type and self.current_token[0] != "EOF":
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = ("EOF", None)
        return

    def eat(self, token_type):
        """Consume the current token when it matches ``token_type``."""
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse(self):
        """Parse the complete token stream into a module AST."""
        module = self.parse_module()
        self.eat("EOF")
        return module

    def parse_module(self):
        """Parse top-level Vulkan/SPIR-V declarations and functions."""
        functions = []
        structs = []
        global_variables = []
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PRECISION":
                self.parse_precision_declaration()
            elif self.current_token[0] == "LAYOUT":
                global_variables.append(self.parse_layout())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "UNIFORM":
                global_variables.append(self.parse_uniform())
            elif (
                (
                    self.current_token[0]
                    in [
                        "VOID",
                        "FLOAT",
                        "INT",
                        "UINT",
                        "BOOL",
                        "VEC2",
                        "VEC3",
                        "VEC4",
                        "MAT2",
                        "MAT3",
                        "MAT4",
                    ]
                    or self.current_token[1] in VALID_DATA_TYPES
                )
                and self.peek(1) == "IDENTIFIER"
                and self.peek(2) == "LPAREN"
            ):
                functions.append(self.parse_function())
            elif (
                self.current_token[0] == "IDENTIFIER"
                or self.current_token[0]
                in [
                    "FLOAT",
                    "INT",
                    "UINT",
                    "BOOL",
                    "VEC2",
                    "VEC3",
                    "VEC4",
                    "MAT2",
                    "MAT3",
                    "MAT4",
                ]
                or self.current_token[1] in VALID_DATA_TYPES
            ):
                global_variables.append(self.parse_variable(self.current_token[1]))
            else:
                self.eat(self.current_token[0])
        return ShaderNode(
            functions=functions,
            structs=structs,
            global_variables=global_variables,
        )

    def parse_precision_declaration(self):
        self.eat("PRECISION")
        if self.current_token[0] in self.PRECISION_QUALIFIER_TOKENS:
            self.eat(self.current_token[0])

        if self.current_token[1] not in VALID_DATA_TYPES:
            raise SyntaxError(f"Unexpected precision type: {self.current_token[1]}")
        self.eat(self.current_token[0])
        self.eat("SEMICOLON")

    def parse_layout(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        bindings = []
        push_constant = False
        if self.current_token[0] == "PUSH_CONSTANT":
            push_constant = True
            self.eat("PUSH_CONSTANT")
        if self.current_token[0] == "COMMA":
            self.eat("COMMA")

        while self.current_token[0] != "RPAREN":
            binding_name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                binding_value = self.current_token[1]
                self.eat("NUMBER")
                bindings.append((binding_name, binding_value))
            else:
                bindings.append((binding_name, None))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")

        declaration_qualifiers = self.parse_layout_declaration_qualifiers()

        layout_type = None
        block_name = None
        if self.current_token[0] in ["IN", "OUT", "UNIFORM", "BUFFER"]:
            layout_type = self.current_token[0]
            self.eat(layout_type)
            declaration_qualifiers.extend(self.parse_layout_declaration_qualifiers())
            if self.current_token[0] == "IDENTIFIER":
                block_name = self.current_token[1]
                self.eat(self.current_token[0])

        data_type = None
        struct_fields = None
        if layout_type in ["UNIFORM", "BUFFER"]:
            if self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                struct_fields = []

                # Parse structured fields within the uniform/push_constant/buffer block
                while self.current_token[0] != "RBRACE":
                    if self.current_token[1] in VALID_DATA_TYPES:
                        field_type = self.current_token[1]
                        self.eat(self.current_token[0])
                    else:
                        raise SyntaxError(
                            "Expected some data type before an identifier"
                        )
                    field_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    field_name += self.parse_array_suffixes_as_text()
                    self.eat("SEMICOLON")
                    struct_fields.append((field_type, field_name))

                self.eat("RBRACE")
                data_type = "struct"
            elif self.current_token[1] in VALID_DATA_TYPES:
                data_type = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    "Expected structured data block after 'uniform' or 'buffer'"
                )
        else:
            if layout_type in ["IN", "OUT"] and self.current_token[0] == "SEMICOLON":
                pass
            elif self.current_token[1] in VALID_DATA_TYPES:
                data_type = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(f"Unexpected type: {self.current_token[1]}")

        variable_name = None
        if self.current_token[0] == "IDENTIFIER":
            variable_name = self.current_token[1]
            self.eat("IDENTIFIER")
            variable_name += self.parse_array_suffixes_as_text()

        self.eat("SEMICOLON")
        return LayoutNode(
            bindings,
            push_constant=push_constant,
            layout_type=layout_type,
            data_type=data_type,
            variable_name=variable_name,
            struct_fields=struct_fields,
            block_name=block_name,
            declaration_qualifiers=declaration_qualifiers,
        )

    def parse_layout_declaration_qualifiers(self):
        qualifiers = []
        while self.current_token[1] in self.LAYOUT_DECLARATION_QUALIFIERS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_push_constant(self):
        self.eat("PUSH_CONSTANT")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.append(self.parse_variable())
        self.eat("RBRACE")
        return PushConstantNode(members)

    def parse_descriptor_set(self):
        self.eat("DESCRIPTOR_SET")
        set_number = self.current_token[1]
        self.eat("NUMBER")
        self.eat("LBRACE")
        bindings = []
        while self.current_token[0] != "RBRACE":
            bindings.append(self.parse_variable())
        self.eat("RBRACE")
        return DescriptorSetNode(set_number, bindings)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] in [
                "VEC2",
                "VEC3",
                "VEC4",
                "IVEC2",
                "IVEC3",
                "IVEC4",
                "UVEC2",
                "UVEC3",
                "UVEC4",
                "FLOAT",
                "INT",
                "UINT",
                "BOOL",
                "MAT2",
                "MAT3",
                "MAT4",
            ]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[1] in VALID_DATA_TYPES:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[0] == "IDENTIFIER":
                type_name = self.current_token[1]
                self.eat("IDENTIFIER")
            else:
                raise SyntaxError(
                    f"Unexpected token in struct member: {self.current_token}"
                )

            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            member_name += self.parse_array_suffixes_as_text()

            self.eat("SEMICOLON")

            members.append(VariableNode(type_name, member_name))

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self):
        return_type = self.current_token[1]
        if self.current_token[1] in VALID_DATA_TYPES:
            self.eat(self.current_token[0])
        else:
            raise SyntaxError(f"Unexpected type: {self.current_token[1]}")
        func_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        body = self.parse_block()
        return FunctionNode(return_type, func_name, params, body)

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            qualifiers = []
            while self.current_token[0] in self.PARAMETER_QUALIFIER_TOKENS:
                qualifiers.append(self.current_token[1])
                self.eat(self.current_token[0])

            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            params.append(VariableNode(vtype, name, qualifiers=qualifiers))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params

    def parse_block(self):
        self.eat("LBRACE")
        statements = []
        while self.current_token[0] != "RBRACE":
            statements.append(self.parse_body())
        self.eat("RBRACE")
        return statements

    def parse_body(self):
        token_type = self.current_token[0]

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
                raise SyntaxError("break used outside loop or switch")
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        elif token_type == "CONTINUE":
            if self.loop_depth == 0:
                raise SyntaxError("continue used outside loop")
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
            elif self.current_token[0] in [
                "EQUALS",
                "PLUS_EQUALS",
                "MINUS_EQUALS",
                "MULTIPLY_EQUALS",
                "DIVIDE_EQUALS",
                "ASSIGN_AND",
                "ASSIGN_OR",
                "ASSIGN_XOR",
                "ASSIGN_MOD",
                "ASSIGN_SHIFT_LEFT",
                "ASSIGN_SHIFT_RIGHT",
            ]:
                op_name = self.current_token[1]
                self.eat(self.current_token[0])
                value = self.parse_expression()
                return AssignmentNode(target, value, op_name)
            else:
                raise SyntaxError(f"Unexpected token in update: {self.current_token[0]}")
        elif self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_update_target())
        elif self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_update_target())
        else:
            raise SyntaxError(f"Unexpected token in update: {self.current_token[0]}")

    def parse_update_target(self):
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(f"Expected update target, got {self.current_token[0]}")

        target = VariableNode("", self.current_token[1])
        self.eat("IDENTIFIER")
        target = self.parse_postfix_suffixes(target)
        if not isinstance(target, (VariableNode, MemberAccessNode, ArrayAccessNode)):
            raise SyntaxError(f"Invalid update target: {type(target).__name__}")
        return target

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        if_condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_block()
        else_body = None
        else_if_chain = []
        while self.current_token[0] == "ELSE" and self.peek(1) == "IF":
            self.eat("ELSE")
            self.eat("IF")
            self.eat("LPAREN")
            else_if_condition = self.parse_expression()
            self.eat("RPAREN")
            else_if_chain.append((else_if_condition, self.parse_block()))
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block()
        return IfNode(
            if_condition,
            if_body,
            else_body,
            else_if_chain=else_if_chain,
        )

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")
        initialization = self.parse_assignment_or_function_call()
        condition = self.parse_expression()
        self.eat("SEMICOLON")
        increment = self.parse_update()
        self.eat("RPAREN")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        return ForNode(initialization, condition, increment, body)

    def parse_variable(self, type_name):
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if type_name:
            name += self.parse_array_suffixes_as_text()
        target = VariableNode(type_name, name)
        if not type_name:
            target = self.parse_postfix_suffixes(target)

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return target

        elif self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()
            self.eat("SEMICOLON")
            return AssignmentNode(target, value)

        elif self.current_token[0] in ("BINARY_AND", "BINARY_OR", "BINARY_XOR"):
            op = self.current_token[0]
            op_symbol = (
                "&" if op == "BINARY_AND" else ("|" if op == "BINARY_OR" else "^")
            )
            self.eat(op)
            right = self.parse_expression()
            self.eat("SEMICOLON")
            return BinaryOpNode(target, op_symbol, right)

        elif self.current_token[0] in (
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
            "EQUAL",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_MOD",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ):
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)
            value = self.parse_expression()
            self.eat("SEMICOLON")
            return BinaryOpNode(target, op_name, value)
        else:
            raise SyntaxError(
                f"Unexpected token after identifier {name}: {self.current_token[0]}"
            )

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected identifier after dot, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_function_call(self, name):
        args = self.parse_call_arguments()
        return FunctionCallNode(name, args)

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return args

    def parse_function_call_or_identifier(self):
        func_name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "LPAREN":
            node = self.parse_function_call(func_name)
        else:
            node = VariableNode("", func_name)
        return self.parse_postfix_suffixes(node)

    def parse_postfix_suffixes(self, node):
        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "LPAREN":
                    node = MethodCallNode(node, member, self.parse_call_arguments())
                else:
                    node = MemberAccessNode(node, member)
                continue

            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                node = ArrayAccessNode(node, index)
                continue

            return node

    def looks_like_member_call_statement(self):
        index = self.pos
        if self.tokens[index][0] != "IDENTIFIER":
            return False

        while index + 2 < len(self.tokens):
            if self.tokens[index + 1][0] != "DOT":
                return False
            if self.tokens[index + 2][0] != "IDENTIFIER":
                return False
            index += 2
            if index + 1 < len(self.tokens) and self.tokens[index + 1][0] == "LPAREN":
                return True

        return False

    def parse_array_suffixes_as_text(self):
        suffix = ""
        while self.current_token[0] == "LBRACKET":
            suffix += "["
            self.eat("LBRACKET")
            while self.current_token[0] != "RBRACKET":
                if self.current_token[0] == "EOF":
                    raise SyntaxError("Unterminated array suffix")
                suffix += str(self.current_token[1])
                self.eat(self.current_token[0])
            self.eat("RBRACKET")
            suffix += "]"
        return suffix

    def parse_primary(self):
        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            value = self.parse_primary()
            return UnaryOpNode("-", value)

        if (
            self.current_token[0] == "BITWISE_NOT"
            or self.current_token[0] == "BINARY_NOT"
        ):
            self.eat(self.current_token[0])
            value = self.parse_primary()
            return UnaryOpNode("~", value)

        if (
            self.current_token[0] == "IDENTIFIER"
            or self.current_token[1] in VALID_DATA_TYPES
        ):
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            if value.endswith("u"):
                value = value[:-1]
            return value
        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            token_type = self.current_token[0]
            op = self.current_token[1]
            self.eat(token_type)
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_assignment_or_function_call(self):
        type_name = ""
        if self.current_token[0] == "IDENTIFIER" and self.peek(1) in [
            "POST_INCREMENT",
            "POST_DECREMENT",
        ]:
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                self.eat("SEMICOLON")
                return UnaryOpNode("POST_INCREMENT", VariableNode("", name))
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                self.eat("SEMICOLON")
                return UnaryOpNode("POST_DECREMENT", VariableNode("", name))
            else:
                raise SyntaxError(
                    f"Unexpected token after identifier: {self.current_token[0]}"
                )
        if self.current_token[0] == "IDENTIFIER" and self.peek(1) == "IDENTIFIER":
            type_name = self.current_token[1]
            self.eat("IDENTIFIER")
        elif self.current_token[1] in VALID_DATA_TYPES:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        if self.current_token[0] == "IDENTIFIER":
            return self.parse_variable(type_name)

    def parse_expression(self):
        left = self.parse_bitwise_expression()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "EQUAL",
            "NOT_EQUAL",
            "AND",
            "OR",
        ]:
            op = self.current_token[0]
            op_symbol = self.current_token[1] if len(self.current_token) > 1 else op
            self.eat(op)
            right = self.parse_bitwise_expression()
            left = BinaryOpNode(left, op_symbol, right)

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)

        return left

    def parse_bitwise_expression(self):
        left = self.parse_additive()
        while self.current_token[0] in [
            "BINARY_AND",
            "BINARY_OR",
            "BINARY_XOR",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_SHIFT_RIGHT",
        ]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_additive()

            op_symbol = (
                "&"
                if op == "BINARY_AND"
                else (
                    "|"
                    if op == "BINARY_OR"
                    else (
                        "^"
                        if op == "BINARY_XOR"
                        else "<<" if op == "BITWISE_SHIFT_LEFT" else ">>"
                    )
                )
            )

            left = BinaryOpNode(left, op_symbol, right)

        return left

    def parse_expression_statement(self):
        expr = self.parse_expression()
        if self.current_token[0] in [
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_MOD",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            op_name = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()
            self.eat("SEMICOLON")
            if op_name == "=":
                return AssignmentNode(expr, value)
            return BinaryOpNode(expr, op_name, value)
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
            body = self.parse_block()
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
        self.breakable_depth += 1
        try:
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                cases.append(self.parse_case_statement())
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated switch statement")
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
            raise SyntaxError(
                f"Expected CASE or DEFAULT in switch, got {self.current_token[0]}"
            )
        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            statements.append(self.parse_body())
        if self.current_token[0] == "EOF":
            raise SyntaxError("Unterminated switch case")
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
        var_type = self.current_token[1]
        if self.current_token[1] in VALID_DATA_TYPES:
            self.eat(self.current_token[0])
        else:
            raise SyntaxError(f"Unexpected type: {self.current_token[1]}")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        name += self.parse_array_suffixes_as_text()
        self.eat("SEMICOLON")
        return UniformNode(var_type, name)

    def parse_unary(self):
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        if self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_unary())
        if self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_unary())
        return self.parse_primary()

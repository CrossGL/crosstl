"""Postfix and primary expressions in the CrossGL parser: member access, calls, generic suffixes, struct constructors, lambda arguments, primary expressions and literals."""

from ..ast import (
    FLOAT_LITERAL_SUFFIX,
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    CommaExpressionNode,
    ConstructorNode,
    CooperativeMatrixOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    MeshOpNode,
    NamedType,
    PointerAccessNode,
    PrimitiveType,
    RayQueryOpNode,
    RayTracingOpNode,
    UnaryOpNode,
    WaveOpNode,
)
from ..integer_literals import integer_literal_parts
from ..math_namespaces import (
    METAL_MATH_NAMESPACE_ANNOTATION,
    metal_math_namespace,
)
from ..refusals import CrossGLSyntaxError
from ..wave_operations import GLSL_SUBGROUP_FUNCTIONS
from .vocabulary import (
    _GENERIC_VECTOR_TYPE_NAMES,
    COOPERATIVE_MATRIX_INTRINSICS,
    MESH_INTRINSICS,
    RAYQUERY_METHODS,
    RAYTRACING_INTRINSICS,
    WAVE_INTRINSICS,
)


class ParserPostfixExpressionMixin:
    """Postfix expression members of ``Parser``."""

    def parse_postfix_expression(self):
        """Parse member, call, index, and postfix unary expressions."""
        left = self.parse_primary_expression()

        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.parse_binding_identifier()
                left = MemberAccessNode(left, member)
            elif self.current_token_is_tuple_field_number():
                # ``t.1`` lexes as ``t`` and the float ``.1``; after an operand it can only be tuple field 1.
                member = str(self.current_token[1])[1:]
                self.eat("FLOAT_NUMBER")
                left = MemberAccessNode(left, member)
            elif self.is_arrow_token():
                self.eat_arrow()
                member = self.parse_binding_identifier()
                left = PointerAccessNode(left, member)
            elif self.current_token[0] == "DOUBLE_COLON":
                self.eat("DOUBLE_COLON")
                member = str(self.current_token[1])
                self.eat(self.current_token[0])
                if isinstance(left, IdentifierNode):
                    left = IdentifierNode(f"{left.name}::{member}")
                else:
                    left = IdentifierNode(
                        f"{self.format_expression_path(left)}::{member}"
                    )
            elif (
                self.current_token[0] == "LBRACKET"
                and isinstance(left, (IdentifierNode, MemberAccessNode))
                and not self.brackets_subscript_a_declared_value(left)
                and self.square_brackets_form_expression_type_arguments()
            ):
                generic_args = self.parse_square_expression_generic_arguments()
                args = ", ".join(self.format_type_argument(arg) for arg in generic_args)
                left = IdentifierNode(f"{self.format_expression_path(left)}[{args}]")
            elif self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                left = ArrayAccessNode(left, index)
            elif (
                self.current_token[0] == "LBRACE"
                and isinstance(left, IdentifierNode)
                and not getattr(self, "suppress_braced_constructor", False)
                and self.is_struct_constructor_brace()
            ):
                left = self.parse_struct_constructor_expression(left.name)
            elif (
                self.current_token[0] == "LESS_THAN"
                and isinstance(left, (MemberAccessNode, PointerAccessNode))
                and self.generic_suffix_is_call_suffix()
            ):
                generic_args = self.parse_generic_arguments(
                    reject_logical_at_top_level=True
                )
                self.eat("LPAREN")
                arguments = self.parse_call_arguments()
                self.eat("RPAREN")
                left = FunctionCallNode(left, arguments, generic_args=generic_args)
            elif (
                self.current_token[0] == "LESS_THAN"
                and isinstance(left, IdentifierNode)
                and (
                    left.name in {"vec2", "vec3", "vec4"}
                    or self.generic_suffix_is_expression_name_suffix()
                )
            ):
                generic_args = self.parse_generic_arguments(
                    reject_logical_at_top_level=True
                )
                if (
                    left.name.rsplit("::", 1)[-1] == "CooperativeMatrix"
                    and self.current_token[0] == "LPAREN"
                ):
                    left = self.parse_cooperative_matrix_constructor(generic_args)
                    continue
                args = ", ".join(self.format_type_argument(arg) for arg in generic_args)
                spelled_name = f"{left.name}<{args}>"
                if (
                    self.current_token[0] == "LBRACE"
                    and not getattr(self, "suppress_braced_constructor", False)
                    and self.is_struct_constructor_brace()
                ):
                    left = self.parse_generic_braced_constructor(
                        NamedType(left.name, generic_args), spelled_name
                    )
                    continue
                left = IdentifierNode(spelled_name)
            elif self.current_token[0] == "LPAREN":
                if isinstance(left, IdentifierNode) and left.name == "lambda":
                    arguments = self.parse_lambda_call_arguments()
                else:
                    self.eat("LPAREN")
                    arguments = self.parse_call_arguments()
                    self.eat("RPAREN")
                if isinstance(left, IdentifierNode):
                    if left.name in WAVE_INTRINSICS:
                        left = WaveOpNode(left.name, arguments)
                    elif left.name in GLSL_SUBGROUP_FUNCTIONS:
                        # CrossGL also accepts the GL_KHR_shader_subgroup spelling of a wave operation.
                        left = WaveOpNode(GLSL_SUBGROUP_FUNCTIONS[left.name], arguments)
                    elif left.name in COOPERATIVE_MATRIX_INTRINSICS:
                        left = self.infer_cooperative_matrix_operation(
                            CooperativeMatrixOpNode(
                                COOPERATIVE_MATRIX_INTRINSICS[left.name],
                                arguments,
                            )
                        )
                    elif left.name in RAYTRACING_INTRINSICS:
                        left = RayTracingOpNode(left.name, arguments)
                    elif left.name in MESH_INTRINSICS:
                        left = MeshOpNode(left.name, arguments)
                    elif metal_math_namespace(left.name) is not None:
                        # Other backends see the plain builtin they always
                        # saw; backends that honor the namespace read it from
                        # the annotation.
                        namespace, operation = metal_math_namespace(left.name)
                        left = FunctionCallNode(
                            IdentifierNode(operation),
                            arguments,
                            annotations={METAL_MATH_NAMESPACE_ANNOTATION: namespace},
                        )
                    else:
                        left = FunctionCallNode(left, arguments)
                elif (
                    isinstance(left, MemberAccessNode)
                    and left.member in RAYQUERY_METHODS
                ):
                    left = RayQueryOpNode(left.member, left.object, arguments)
                else:
                    left = FunctionCallNode(left, arguments)
            elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                left = UnaryOpNode(op, left, is_postfix=True)
            else:
                break

        return left

    def parse_call_arguments(self):
        """Parse comma-separated call arguments without accepting stalled input."""
        arguments = []
        while self.current_token[0] != "RPAREN":
            if self.current_token[0] == "EOF":
                raise CrossGLSyntaxError("Unterminated function call argument list")

            arguments.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            if self.current_token[0] != "RPAREN":
                token_type, token_value = self.current_token
                raise CrossGLSyntaxError(
                    "Expected ',' or ')' after function argument, "
                    f"got {token_type} {token_value!r}"
                )

        return arguments

    def parse_square_expression_generic_arguments(self):
        """Parse square-bracket specialization arguments in expression position."""
        self.eat("LBRACKET")
        args = []

        while self.current_token[0] != "RBRACKET":
            if self.current_token[0] == "RANGE" and self.peek()[0] == "DOT":
                self.eat("RANGE")
                self.eat("DOT")
                args.append(IdentifierNode("..."))
            elif self.current_token[0] == "MULTIPLY":
                self.eat("MULTIPLY")
                args.append(IdentifierNode("*"))
            elif (
                self.current_token_is_binding_identifier() and self.peek()[0] == "COLON"
            ):
                args.append(self.parse_square_generic_argument())
            elif (
                self.current_token_is_binding_identifier()
                and self.peek()[0] == "EQUALS"
            ):
                name = self.parse_binding_identifier()
                self.eat("EQUALS")
                args.append(
                    AssignmentNode(IdentifierNode(name), self.parse_expression())
                )
            else:
                args.append(self.parse_expression())

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RBRACKET")
        return args

    def brackets_subscript_a_declared_value(self, left):
        """Whether ``left[...]`` indexes a declared variable rather than specializing a name.

        ``t[j = k + 1]`` and ``t[i, j]`` look like specialization arguments (``name = value``, a comma) to
        :meth:`square_brackets_form_expression_type_arguments`, but a declared variable cannot be specialized:
        its brackets are a subscript whose index is an assignment or a comma expression. A call after the
        brackets keeps the specialization reading.
        """
        root = left
        while isinstance(root, MemberAccessNode):
            root = root.object
        if not isinstance(root, IdentifierNode):
            return False
        name = root.name
        if (
            name in self.declared_type_names
            or name in self.active_generic_parameter_names
            or name in self.parameterized_value_names
        ):
            return False
        if not any(name in scope for scope in self.value_type_scopes):
            return False
        return self.token_after_matching_bracket() != "LPAREN"

    def token_after_matching_bracket(self):
        """Type of the token after the ``]`` that closes the current ``[``."""
        depth = 0
        index = self.pos
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    return (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
            index += 1
        return "EOF"

    def square_brackets_form_expression_type_arguments(self):
        """Return whether expression ``[...]`` is a specialization suffix."""
        if self.current_token[0] != "LBRACKET":
            return False

        index = self.pos + 1
        bracket_depth = 1
        paren_depth = 0
        brace_depth = 0
        saw_generic_separator = False
        ternary_depth = 0

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET":
                bracket_depth -= 1
                if bracket_depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return saw_generic_separator or next_type == "LPAREN"
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth:
                paren_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth:
                brace_depth -= 1
            elif (
                bracket_depth == 1
                and paren_depth == 0
                and brace_depth == 0
                and token_type in {"COMMA", "EQUALS"}
            ):
                saw_generic_separator = True
            elif (
                bracket_depth == 1
                and paren_depth == 0
                and brace_depth == 0
                and token_type == "QUESTION"
            ):
                ternary_depth += 1
            elif (
                bracket_depth == 1
                and paren_depth == 0
                and brace_depth == 0
                and token_type == "COLON"
            ):
                if ternary_depth:
                    ternary_depth -= 1
                else:
                    saw_generic_separator = True

            index += 1

        return False

    def generic_suffix_is_expression_name_suffix(self):
        """Return whether ``<...>`` belongs to an identifier expression name."""
        if self.current_token[0] != "LESS_THAN":
            return False

        saved_pos = self.pos
        saved_token = self.current_token
        token_mark = self.mark_token_edits()
        try:
            generic_args = self.parse_generic_arguments(
                reject_logical_at_top_level=True, validate_arguments=False
            )
            if self.current_token[0] == "LBRACE":
                # ``vec<T, 2>{x, y}``: a braced constructor of a generic type, as MSL and HLSL write one. A
                # relational chain is never followed by ``{``, but a single symbolic argument must name a type.
                return self.generic_member_arguments_are_unambiguous(generic_args)
            return self.current_token[0] in {
                "COMMA",
                "DOT",
                "LBRACKET",
                "LPAREN",
                "RBRACE",
                "RBRACKET",
                "RPAREN",
                "SEMICOLON",
            }
        except SyntaxError:
            return False
        finally:
            self.undo_token_edits(token_mark)
            self.pos = saved_pos
            self.current_token = saved_token

    def generic_suffix_is_call_suffix(self):
        """Return whether ``<...>`` is followed immediately by a call."""
        if self.current_token[0] != "LESS_THAN":
            return False

        saved_pos = self.pos
        saved_token = self.current_token
        token_mark = self.mark_token_edits()
        try:
            generic_args = self.parse_generic_arguments(
                reject_logical_at_top_level=True, validate_arguments=False
            )
            return self.current_token[0] == "LPAREN" and (
                self.generic_member_arguments_are_unambiguous(generic_args)
            )
        except SyntaxError:
            return False
        finally:
            self.undo_token_edits(token_mark)
            self.pos = saved_pos
            self.current_token = saved_token

    def generic_member_arguments_are_unambiguous(self, generic_args):
        """Distinguish a single symbolic generic from a relational chain."""
        if len(generic_args) != 1:
            return True
        argument = generic_args[0]
        if not isinstance(argument, NamedType) or argument.generic_args:
            return True
        return argument.name in (
            self.active_generic_parameter_names | self.declared_type_names
        )

    def format_expression_path(self, expression):
        if isinstance(expression, IdentifierNode):
            return expression.name
        if isinstance(expression, MemberAccessNode):
            return f"{self.format_expression_path(expression.object_expr)}.{expression.member}"
        return self.format_type_argument(expression)

    def parse_struct_constructor_expression(self, type_name):
        self.eat("LBRACE")
        arguments = []
        named_arguments = {}

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue

            if self.peek()[0] == "COLON":
                field_name = str(self.current_token[1])
                self.eat(self.current_token[0])
                self.eat("COLON")
                named_arguments[field_name] = self.parse_expression()
            elif self.is_constructor_shorthand_field():
                field_name = str(self.current_token[1])
                self.eat(self.current_token[0])
                named_arguments[field_name] = IdentifierNode(field_name)
            else:
                arguments.append(self.parse_expression())

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RBRACE")
        constructor_type = (
            NamedType(type_name) if isinstance(type_name, str) else type_name
        )
        node = ConstructorNode(constructor_type, arguments, named_arguments)
        node.is_braced_constructor = True
        return node

    def parse_generic_braced_constructor(self, generic_type, spelled_name):
        """Parse ``Type<args>{...}``, the braced constructor of a generic type, after its type arguments.

        A vector's braced list initializes its components in order, as its constructor does (MSL section 2.2.3), so
        ``vec<T, 2>{x, y}`` reads as the call ``vec<T, 2>(x, y)``; any other type's is an aggregate constructor
        that names the type as a declaration does.
        """
        constructor = self.parse_struct_constructor_expression(generic_type)
        if (
            generic_type.name in _GENERIC_VECTOR_TYPE_NAMES
            and constructor.arguments
            and not constructor.named_arguments
        ):
            return FunctionCallNode(IdentifierNode(spelled_name), constructor.arguments)
        return constructor

    def is_constructor_shorthand_field(self):
        """Return whether the current token is a braced-constructor field shorthand."""
        field_name = self.current_token[1]
        return (
            isinstance(field_name, str)
            and field_name.isidentifier()
            and self.peek()[0] in ["COMMA", "RBRACE"]
        )

    def is_struct_constructor_brace(self):
        """Return whether the current ``{`` looks like a braced constructor."""
        if self.current_token[0] != "LBRACE":
            return False

        next_type = self.peek()[0]
        if next_type == "RBRACE":
            return True

        offset = 1
        depth = 1
        while True:
            token_type = self.peek(offset)[0]
            if token_type == "EOF":
                return False
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                if depth == 1:
                    return False
                depth -= 1
            elif depth == 1:
                if token_type in ["COLON", "COMMA"]:
                    return True
                if token_type in ["SEMICOLON", "FAT_ARROW"]:
                    return False
            offset += 1

    def parse_lambda_call_arguments(self):
        """Parse CrossGL's compact lambda syntax without splitting typed params."""
        self.eat("LPAREN")
        arguments = []
        raw_tokens = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token

            if (
                token_type == "RPAREN"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                break

            if (
                token_type == "COMMA"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and angle_depth == 0
            ):
                arguments.append(self.raw_lambda_argument(raw_tokens))
                raw_tokens = []
                self.eat("COMMA")
                continue

            raw_tokens.append((token_type, token_value))

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth > 0:
                brace_depth -= 1
            elif token_type == "LESS_THAN":
                angle_depth += 1
            elif token_type == "GREATER_THAN" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "BITWISE_SHIFT_RIGHT" and angle_depth > 0:
                angle_depth = max(0, angle_depth - 2)

            self.eat(token_type)

        if raw_tokens:
            arguments.append(self.raw_lambda_argument(raw_tokens))

        self.eat("RPAREN")
        return arguments

    def raw_lambda_argument(self, tokens):
        return IdentifierNode(self.format_raw_lambda_tokens(tokens))

    def format_raw_lambda_tokens(self, tokens):
        compact = ""
        format_angle_depth = 0

        for token_type, token_value in tokens:
            value = str(token_value) if token_value is not None else ""
            if not value:
                continue

            if not compact:
                compact = value
                continue

            if value == "}":
                if compact[-1:] not in {" ", "{"}:
                    compact += " "
                compact += value
            elif value == "<":
                previous_word = compact.rsplit(maxsplit=1)[-1]
                if previous_word and (
                    previous_word[0].isupper()
                    or previous_word in {"Vec", "Option", "Result"}
                ):
                    compact += value
                    format_angle_depth += 1
                else:
                    compact += f" {value} "
            elif value == ">":
                if format_angle_depth > 0:
                    compact += value
                    format_angle_depth -= 1
                else:
                    compact += f" {value} "
            elif value == ">>":
                if format_angle_depth > 1:
                    compact += value
                    format_angle_depth -= 2
                else:
                    compact += f" {value} "
            elif value == "{":
                if compact[-1:] not in {" ", "(", "[", "<"}:
                    compact += " "
                compact += value
            elif value in {")", "]", ",", ";", ":"}:
                compact += value
            elif value in {"(", "["}:
                previous_word = compact.rsplit(maxsplit=1)[-1]
                if value == "(" and previous_word in {
                    "return",
                    "if",
                    "while",
                    "for",
                    "switch",
                }:
                    compact += " "
                elif value == "[" and compact.endswith("]"):
                    compact += " "
                if compact[-1:].isalnum() or compact[-1:] == "_":
                    compact += value
                else:
                    compact += value
            elif compact[-1:] in {"(", "[", "<"}:
                compact += value
            elif compact[-1:] == "{":
                compact += f" {value}"
            elif value in {".", "::"} or compact.endswith(("::", ".")):
                compact += value
            else:
                separator = "" if compact.endswith(" ") else " "
                compact += f"{separator}{value}"

        return compact

    def parse_primary_expression(self):
        """Parse identifiers, literals, parenthesized expressions, and arrays."""
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "def"
            and self.peek()[0] in {"LBRACKET", "LPAREN"}
        ):
            return self.parse_callable_type()

        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return IdentifierNode(name)

        elif self.current_token[0] == "BOOLEAN_LITERAL":
            return self.parse_literal()

        elif self.current_token[0] in [
            "NUMBER",
            "FLOAT_NUMBER",
            "HEX_NUMBER",
            "BIN_NUMBER",
            "OCT_NUMBER",
        ]:
            return self.parse_literal()

        elif self.current_token[0] in ["STRING_LITERAL", "CHAR_LITERAL"]:
            return self.parse_literal()

        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            if self.current_token[0] == "RPAREN":
                self.eat("RPAREN")
                return ArrayLiteralNode([])
            expr = self.parse_expression()
            if self.current_token[0] == "COMMA":
                # ``(a, b)`` is the comma operator; a trailing comma, ``(a, b,)``, keeps the tuple form.
                elements = [expr]
                trailing_comma = False
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_token[0] == "RPAREN":
                        trailing_comma = True
                        break
                    elements.append(self.parse_expression())
                self.eat("RPAREN")
                if trailing_comma:
                    return ArrayLiteralNode(elements)
                return CommaExpressionNode(elements)
            self.eat("RPAREN")
            return expr

        elif self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            elements = []
            while self.current_token[0] != "RBRACKET":
                elements.append(self.parse_expression())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_token[0] == "RBRACKET":
                        break
                    continue
                break
            self.eat("RBRACKET")
            return ArrayLiteralNode(elements)

        elif self.current_token[0] == "MATCH":
            if self.peek()[0] in {
                "ASSIGN_ADD",
                "ASSIGN_AND",
                "ASSIGN_DIV",
                "ASSIGN_MOD",
                "ASSIGN_MUL",
                "ASSIGN_OR",
                "ASSIGN_SHIFT_LEFT",
                "ASSIGN_SHIFT_RIGHT",
                "ASSIGN_SUB",
                "ASSIGN_XOR",
                "BITWISE_AND",
                "BITWISE_OR",
                "COMMA",
                "DIVIDE",
                "DOT",
                "EQUAL",
                "EQUALS",
                "GREATER_THAN",
                "LBRACKET",
                "LESS_THAN",
                "LOGICAL_AND",
                "LOGICAL_OR",
                "LPAREN",
                "MINUS",
                "MOD",
                "MULTIPLY",
                "NOT_EQUAL",
                "PLUS",
                "RBRACE",
                "RBRACKET",
                "RPAREN",
                "SEMICOLON",
            }:
                name = self.current_token[1]
                self.eat("MATCH")
                return IdentifierNode(name)
            return self.parse_match_statement()

        elif self.current_token[0] == "LBRACE":
            return self.parse_array_literal()

        else:
            name = str(self.current_token[1]) if self.current_token[1] else ""
            if not name.isidentifier():
                # A keyword spelled like a name (``vec4`` in ``vec4(x)``) is read as one; punctuation and the end of
                # input start no expression, and turning them into identifiers would emit them as values.
                self.reject_unexpected_token("an expression")
            self.eat(self.current_token[0])
            return IdentifierNode(name)

    def parse_array_literal(self):
        """Parse a braced array literal expression."""
        self.eat("LBRACE")
        elements = []

        while self.current_token[0] != "RBRACE":
            elements.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RBRACE":
                    break
            elif self.current_token[0] != "RBRACE":
                break

        self.eat("RBRACE")
        return ArrayLiteralNode(elements)

    def parse_literal(self):
        """Parse a scalar literal token into a typed literal node."""
        token_type, value = self.current_token

        if token_type == "NUMBER":
            self.eat("NUMBER")
            digits, literal_type = self.parse_integer_literal_parts(value)
            return LiteralNode(int(digits), literal_type)

        elif token_type == "FLOAT_NUMBER":
            self.eat("FLOAT_NUMBER")
            literal_text = value.rstrip("fF")
            literal_value = (
                float.fromhex(literal_text)
                if literal_text.lower().startswith("0x")
                else float(literal_text)
            )
            literal = LiteralNode(literal_value, PrimitiveType("float"))
            if literal_text != value:
                # ``0.1f`` is a binary32 value; an unsuffixed ``0.1`` is its exact decimal value, rounded once
                # to the type its context requires (FLOAT_LITERAL_SUFFIX).
                literal.add_annotation(FLOAT_LITERAL_SUFFIX, "f")
            return literal

        elif token_type == "BOOLEAN_LITERAL":
            self.eat("BOOLEAN_LITERAL")
            return LiteralNode(value == "true", PrimitiveType("bool"))

        elif token_type == "HEX_NUMBER":
            self.eat("HEX_NUMBER")
            digits, literal_type = self.parse_integer_literal_parts(value)
            return LiteralNode(int(digits, 16), literal_type)

        elif token_type == "BIN_NUMBER":
            self.eat("BIN_NUMBER")
            digits, literal_type = self.parse_integer_literal_parts(value)
            return LiteralNode(int(digits, 2), literal_type)

        elif token_type == "OCT_NUMBER":
            self.eat("OCT_NUMBER")
            digits, literal_type = self.parse_integer_literal_parts(value)
            return LiteralNode(int(digits, 8), literal_type)

        elif token_type == "STRING_LITERAL":
            self.eat("STRING_LITERAL")
            return LiteralNode(value[1:-1], PrimitiveType("string"))

        elif token_type == "CHAR_LITERAL":
            self.eat("CHAR_LITERAL")
            return LiteralNode(value[1:-1], PrimitiveType("char"))

        else:
            self.eat(token_type)
            return LiteralNode(value, PrimitiveType("unknown"))

    def parse_integer_literal_parts(self, value):
        """Retain an integer literal's source width and signedness."""
        digits, type_name = integer_literal_parts(value)
        return digits, PrimitiveType(type_name)

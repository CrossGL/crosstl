"""Type arguments in the CrossGL parser: callable types, square-bracket and angle-bracket generic parameters and arguments, qualified identifiers, type start tokens and type argument formatting."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    FunctionType,
    GenericParameterNode,
    IdentifierNode,
    MatrixType,
    MemberAccessNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    TernaryOpNode,
    UnaryOpNode,
)
from ..refusals import CrossGLSyntaxError
from .vocabulary import WGSL_STYLE_SCALAR_SPELLINGS


class ParserTypeArgumentMixin:
    """Type argument members of ``Parser``."""

    def parse_callable_type(self):
        """Parse Mojo-style function type syntax emitted by reverse codegen."""
        self.eat("IDENTIFIER")

        generic_params = []
        if self.current_token[0] == "LBRACKET":
            generic_params = self.parse_square_generic_arguments()
            generic_params = [
                param
                for param in generic_params
                if not (isinstance(param, IdentifierNode) and param.name == "*")
            ]

        self.eat("LPAREN")
        param_types = []
        while self.current_token[0] != "RPAREN":
            if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
                self.parse_binding_identifier()
                self.eat("COLON")
            param_types.append(self.parse_type())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break
        self.eat("RPAREN")

        effects = []
        while self.current_token[0] in {"IDENTIFIER", "RAISES"}:
            effect = str(self.current_token[1])
            if effect not in {"capturing", "raises", "thin", "async"}:
                break
            effects.append(effect)
            self.eat(self.current_token[0])
            if self.current_token[0] == "LBRACKET":
                self.skip_balanced_brackets()

        return_type = PrimitiveType("void")
        if self.is_arrow_token():
            self.eat_arrow()
            return_type = self.parse_type()

        callable_type = FunctionType(return_type, param_types)
        callable_type.annotations["generic_params"] = generic_params
        callable_type.annotations["effects"] = effects
        return callable_type

    def square_brackets_form_type_arguments(self):
        """Return whether ``[...]`` is a generic argument list, not an array size."""
        if self.current_token[0] != "LBRACKET":
            return False

        index = self.pos + 1
        depth = 1
        saw_generic_separator = False
        saw_nested_brackets = False
        ternary_depth = 0

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                depth += 1
                saw_nested_brackets = True
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return (
                        saw_generic_separator
                        or saw_nested_brackets
                        or next_type == "LPAREN"
                    )
            elif depth == 1 and token_type in {"COMMA", "EQUALS"}:
                saw_generic_separator = True
            elif depth == 1 and token_type == "QUESTION":
                ternary_depth += 1
            elif depth == 1 and token_type == "COLON":
                if ternary_depth:
                    ternary_depth -= 1
                else:
                    saw_generic_separator = True
            elif (
                depth == 1
                and token_type == "RANGE"
                and index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "DOT"
            ):
                saw_generic_separator = True

            index += 1

        return False

    def square_brackets_form_member_type_arguments(self):
        """Return whether a post-generic member suffix has type arguments."""
        if self.current_token[0] != "LBRACKET":
            return False

        index = self.pos + 1
        depth = 1

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    return False
            elif depth == 1 and token_type in {"DOT", "DOUBLE_COLON"}:
                return True

            index += 1

        return False

    def parse_square_generic_arguments(self):
        """Parse Mojo-style square-bracket type arguments."""
        self.eat("LBRACKET")
        args = []

        while self.current_token[0] != "RBRACKET":
            arg = self.parse_square_generic_argument()
            self.format_type_argument(arg)
            args.append(arg)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RBRACKET")
        return args

    def parse_square_generic_argument(self):
        """Parse one square-bracket generic argument or parameter declaration."""
        if self.current_token[0] == "RANGE" and self.peek()[0] == "DOT":
            self.eat("RANGE")
            self.eat("DOT")
            return IdentifierNode("...")

        if self.current_token[0] == "MULTIPLY":
            self.eat("MULTIPLY")
            return IdentifierNode("*")

        if self.current_token[0] == "DIVIDE":
            self.eat("DIVIDE")
            return IdentifierNode("/")

        if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
            name = self.parse_binding_identifier()
            self.eat("COLON")

            constraints = [self.parse_type()]
            while self.current_token[0] in {"PLUS", "BITWISE_AND"}:
                self.eat(self.current_token[0])
                constraints.append(self.parse_type())

            default_type = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_type = self.parse_expression()

            return GenericParameterNode(
                name=name,
                constraints=constraints,
                default_type=default_type,
            )

        if self.current_token_is_binding_identifier() and self.peek()[0] == "EQUALS":
            name = self.parse_binding_identifier()
            self.eat("EQUALS")
            return AssignmentNode(IdentifierNode(name), self.parse_expression())

        if self.current_token[0] == "IDENTIFIER" and self.peek()[0] in {
            "DOT",
            "DOUBLE_COLON",
            "LPAREN",
        }:
            return self.parse_expression()

        if self.square_generic_argument_forms_expression():
            return self.parse_expression()

        if self.current_token_starts_type():
            return self.parse_type()

        if self.current_token[0] in {
            "BOOLEAN_LITERAL",
            "NUMBER",
            "FLOAT_NUMBER",
            "HEX_NUMBER",
            "BIN_NUMBER",
            "OCT_NUMBER",
            "STRING_LITERAL",
            "CHAR_LITERAL",
        }:
            return self.parse_literal()

        return self.parse_expression()

    def square_generic_argument_forms_expression(self):
        """Return whether one square generic arg contains expression operators."""
        expression_operators = {
            "PLUS",
            "MINUS",
            "MULTIPLY",
            "DIVIDE",
            "MOD",
            "LOGICAL_AND",
            "LOGICAL_OR",
            "BITWISE_AND",
            "BITWISE_OR",
            "BITWISE_XOR",
            "SHIFT_LEFT",
            "SHIFT_RIGHT",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "EQUAL",
            "NOT_EQUAL",
            "RANGE",
            "RANGE_INCLUSIVE",
        }
        opener_to_closer = {
            "LPAREN": "RPAREN",
            "LBRACKET": "RBRACKET",
            "LBRACE": "RBRACE",
        }
        closers = set(opener_to_closer.values())
        stack = []
        index = self.pos

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if not stack and token_type in {"COMMA", "RBRACKET"}:
                return False
            if token_type in opener_to_closer:
                stack.append(opener_to_closer[token_type])
            elif token_type in closers:
                if stack and token_type == stack[-1]:
                    stack.pop()
                elif not stack:
                    return False
            elif not stack and token_type in expression_operators:
                return True
            index += 1

        return False

    def skip_balanced_brackets(self):
        """Consume a square-bracket payload used by type effects."""
        self.eat("LBRACKET")
        depth = 1
        while depth and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACKET":
                depth += 1
            elif self.current_token[0] == "RBRACKET":
                depth -= 1

            token_type = self.current_token[0]
            self.eat(token_type)

    def parse_generic_parameters(self):
        """Parse generic parameter declarations after ``<``."""
        self.eat("LESS_THAN")
        params = []

        while not self.current_token_is_type_greater_than():
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            constraints = []
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                constraints.append(self.parse_type())

                while self.current_token[0] == "PLUS":
                    self.eat("PLUS")
                    constraints.append(self.parse_type())

            default_type = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_type = self.parse_type()

            params.append(
                GenericParameterNode(
                    name=name,
                    constraints=constraints,
                    default_type=default_type,
                )
            )

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat_type_greater_than()
        return params

    def parse_generic_arguments(
        self, reject_logical_at_top_level=False, validate_arguments=True
    ):
        """Parse generic type arguments after ``<``."""
        self.eat("LESS_THAN")
        args = []

        while not self.current_token_is_type_greater_than():
            if self.current_token[0] == "EOF":
                raise CrossGLSyntaxError("Unterminated generic argument list")

            arg = self.parse_generic_argument(
                reject_logical_at_top_level=reject_logical_at_top_level
            )
            if validate_arguments:
                self.format_type_argument(arg)
            args.append(arg)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif not self.current_token_is_type_greater_than():
                raise CrossGLSyntaxError(
                    "Expected ',' or '>' after generic argument, "
                    f"got {self.current_token[0]} '{self.current_token[1]}'"
                )

        self.eat_type_greater_than()
        return args

    def parse_generic_argument(self, reject_logical_at_top_level=False):
        """Parse one angle-bracket type or value argument."""
        start_pos = self.pos
        saved_token = self.current_token
        token_mark = self.mark_token_edits()
        argument = None

        try:
            if self.current_token_is_parameter_qualifier() and self.peek()[0] not in {
                "COMMA",
                "GREATER_THAN",
                "BITWISE_SHIFT_RIGHT",
            }:
                qualifiers, resource_qualifiers = self.partition_resource_qualifiers(
                    self.parse_parameter_qualifiers()
                )
                if not self.current_token_starts_type():
                    raise CrossGLSyntaxError(
                        "Expected a type after generic argument qualifiers"
                    )
                argument = self.parse_type()
                self.apply_pointer_resource_contract(
                    argument, qualifiers, resource_qualifiers
                )
                argument.qualifiers = qualifiers
            elif self.current_token_starts_qualified_identifier():
                argument = IdentifierNode(self.parse_qualified_identifier())
            elif self.current_token_starts_type():
                argument = self.parse_type()
            elif self.current_token[0] in {
                "BOOLEAN_LITERAL",
                "NUMBER",
                "FLOAT_NUMBER",
                "HEX_NUMBER",
                "BIN_NUMBER",
                "OCT_NUMBER",
                "STRING_LITERAL",
                "CHAR_LITERAL",
            }:
                argument = self.parse_literal()
        except SyntaxError:
            argument = None

        if argument is not None and self.current_token[0] in {
            "COMMA",
            "GREATER_THAN",
            "BITWISE_SHIFT_RIGHT",
        }:
            return argument

        self.undo_token_edits(token_mark)
        self.pos = start_pos
        self.current_token = saved_token
        argument = self.parse_expression(
            stop_at_generic_close=True,
            reject_logical_at_top_level=reject_logical_at_top_level,
        )

        if self.current_token[0] == "EOF":
            raise CrossGLSyntaxError("Unterminated generic argument list")
        if self.pos == start_pos or self.current_token[0] not in {
            "COMMA",
            "GREATER_THAN",
            "BITWISE_SHIFT_RIGHT",
        }:
            raise CrossGLSyntaxError(
                "Expected ',' or '>' after generic argument, "
                f"got {self.current_token[0]} '{self.current_token[1]}'"
            )
        return argument

    def current_token_starts_qualified_identifier(self):
        """Return whether current tokens form a ``namespace::name`` value."""
        return (
            self.current_token[0] == "IDENTIFIER" and self.peek()[0] == "DOUBLE_COLON"
        )

    def parse_qualified_identifier(self):
        """Parse a double-colon-qualified identifier into a string."""
        segments = [str(self.current_token[1])]
        self.eat("IDENTIFIER")

        while self.current_token[0] == "DOUBLE_COLON":
            self.eat("DOUBLE_COLON")
            segments.append(str(self.current_token[1]))
            self.eat(self.current_token[0])

        return "::".join(segments)

    def current_token_starts_type(self):
        """Return whether the current token can start a type expression."""
        return self.current_token[0] in self.type_start_tokens()

    def type_start_tokens(self):
        """Return token kinds that can begin a type expression."""
        return {
            "BUFFER",
            "BOOL",
            "I8",
            "I16",
            "I32",
            "I64",
            "U8",
            "U16",
            "U32",
            "U64",
            "F16",
            "BFLOAT16",
            "F32",
            "F64",
            "INT",
            "UINT",
            "FLOAT",
            "DOUBLE",
            "HALF",
            "CHAR",
            "STRING",
            "VOID",
            "TEXTURE1D",
            "TEXTURE2D",
            "TEXTURE3D",
            "TEXTURECUBE",
            "TEXTURE2DARRAY",
            "VEC2",
            "VEC3",
            "VEC4",
            "IVEC2",
            "IVEC3",
            "IVEC4",
            "UVEC2",
            "UVEC3",
            "UVEC4",
            "DVEC2",
            "DVEC3",
            "DVEC4",
            "BVEC2",
            "BVEC3",
            "BVEC4",
            "MAT2",
            "MAT3",
            "MAT4",
            "MAT2X2",
            "MAT2X3",
            "MAT2X4",
            "MAT3X2",
            "MAT3X3",
            "MAT3X4",
            "MAT4X2",
            "MAT4X3",
            "MAT4X4",
            "DMAT2",
            "DMAT3",
            "DMAT4",
            "DMAT2X2",
            "DMAT2X3",
            "DMAT2X4",
            "DMAT3X2",
            "DMAT3X3",
            "DMAT3X4",
            "DMAT4X2",
            "DMAT4X3",
            "DMAT4X4",
            "SAMPLER",
            "SAMPLER1D",
            "SAMPLER1DARRAY",
            "SAMPLER2D",
            "SAMPLER2DARRAY",
            "SAMPLER3D",
            "SAMPLERCUBE",
            "SAMPLERCUBEARRAY",
            "SAMPLER2DMS",
            "SAMPLER2DMSARRAY",
            "SAMPLER1DSHADOW",
            "SAMPLER2DSHADOW",
            "SAMPLER2DARRAYSHADOW",
            "SAMPLERCUBESHADOW",
            "SAMPLERCUBEARRAYSHADOW",
            "ISAMPLER1D",
            "ISAMPLER1DARRAY",
            "ISAMPLER2D",
            "ISAMPLER2DARRAY",
            "ISAMPLER3D",
            "ISAMPLERCUBE",
            "ISAMPLERCUBEARRAY",
            "ISAMPLER2DMS",
            "ISAMPLER2DMSARRAY",
            "USAMPLER1D",
            "USAMPLER1DARRAY",
            "USAMPLER2D",
            "USAMPLER2DARRAY",
            "USAMPLER3D",
            "USAMPLERCUBE",
            "USAMPLERCUBEARRAY",
            "USAMPLER2DMS",
            "USAMPLER2DMSARRAY",
            "IIMAGE1D",
            "IIMAGE1DARRAY",
            "IIMAGE2D",
            "IIMAGE2DARRAY",
            "IIMAGE3D",
            "IIMAGECUBE",
            "IIMAGECUBEARRAY",
            "IIMAGE2DMS",
            "IIMAGE2DMSARRAY",
            "UIMAGE1D",
            "UIMAGE1DARRAY",
            "UIMAGE2D",
            "UIMAGE2DARRAY",
            "UIMAGE3D",
            "UIMAGECUBE",
            "UIMAGECUBEARRAY",
            "UIMAGE2DMS",
            "UIMAGE2DMSARRAY",
            "IMAGE1D",
            "IMAGE1DARRAY",
            "IMAGE2D",
            "IMAGE2DARRAY",
            "IMAGE3D",
            "IMAGECUBE",
            "IMAGECUBEARRAY",
            "IMAGE2DMS",
            "IMAGE2DMSARRAY",
            "IDENTIFIER",
            "LBRACKET",
        }

    def vector_element_type_from_generic(self, type_node):
        """Resolve a vector generic argument to a primitive element type."""
        type_name = self.format_type_argument(type_node)
        # The argument names a CrossGL type: a scalar, a type alias or a generic parameter. ``f32``, ``f64``, ``i32``
        # and ``u32`` are other spellings of four scalars.
        if type_name in WGSL_STYLE_SCALAR_SPELLINGS:
            type_name = WGSL_STYLE_SCALAR_SPELLINGS[type_name]
        return PrimitiveType(type_name)

    def format_type_argument(self, type_node):
        """Format a parsed type node back into a compact type argument string."""
        if isinstance(type_node, AssignmentNode):
            left = self.format_type_argument(type_node.left)
            right = self.format_type_argument(type_node.right)
            return f"{left} = {right}"
        if isinstance(type_node, ArrayLiteralNode):
            elements = ", ".join(
                self.format_type_argument(element) for element in type_node.elements
            )
            return f"[{elements}]"
        if hasattr(type_node, "value"):
            value = type_node.value
            if isinstance(value, bool):
                return str(value).lower()
            literal_type = getattr(
                getattr(type_node, "literal_type", None), "name", None
            )
            if literal_type == "uint" and isinstance(value, int):
                # ``5u`` stays unsigned: a value argument binds a parameter of the literal's type.
                return f"{value}u"
            return str(value)
        if isinstance(type_node, ArrayAccessNode):
            array = self.format_type_argument(type_node.array)
            index = self.format_type_argument(type_node.index)
            return f"{array}[{index}]"
        if isinstance(type_node, BinaryOpNode):
            # A nested operation keeps its grouping: ``(N + 1) * 2`` is not ``N + 1 * 2``.
            left, right = (
                (
                    f"({self.format_type_argument(operand)})"
                    if isinstance(operand, (BinaryOpNode, TernaryOpNode))
                    else self.format_type_argument(operand)
                )
                for operand in (type_node.left, type_node.right)
            )
            return f"{left} {type_node.operator} {right}"
        if isinstance(type_node, FunctionCallNode):
            function = self.format_type_argument(type_node.function)
            generic_suffix = ""
            if getattr(type_node, "generic_args", None):
                generic_args = ", ".join(
                    self.format_type_argument(argument)
                    for argument in type_node.generic_args
                )
                generic_suffix = f"<{generic_args}>"
            args = ", ".join(
                self.format_type_argument(argument) for argument in type_node.arguments
            )
            return f"{function}{generic_suffix}({args})"
        if isinstance(type_node, TernaryOpNode):
            condition = self.format_type_argument(type_node.condition)
            true_expr = self.format_type_argument(type_node.true_expr)
            false_expr = self.format_type_argument(type_node.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        if isinstance(type_node, UnaryOpNode):
            operand = self.format_type_argument(type_node.operand)
            return (
                f"{operand}{type_node.operator}"
                if type_node.is_postfix
                else f"{type_node.operator}{operand}"
            )
        if isinstance(type_node, MemberAccessNode):
            return (
                f"{self.format_type_argument(type_node.object_expr)}.{type_node.member}"
            )
        if isinstance(type_node, PointerType):
            return f"{self.format_type_argument(type_node.pointee_type)}*"
        if isinstance(type_node, ReferenceType):
            suffix = "&mut" if type_node.is_mutable else "&"
            return f"{self.format_type_argument(type_node.referenced_type)}{suffix}"
        if isinstance(type_node, FunctionType):
            params = ", ".join(
                self.format_type_argument(param_type)
                for param_type in type_node.param_types
            )
            return_type = self.format_type_argument(type_node.return_type)
            return f"def({params}) -> {return_type}"
        if isinstance(type_node, GenericParameterNode):
            constraints = " & ".join(
                self.format_type_argument(constraint)
                for constraint in type_node.constraints
            )
            result = (
                f"{type_node.name}:{constraints}" if constraints else type_node.name
            )
            if type_node.default_type is not None:
                result = (
                    f"{result} = {self.format_type_argument(type_node.default_type)}"
                )
            return result
        if isinstance(type_node, MatrixType):
            element_type = self.format_type_argument(type_node.element_type)
            dims = (
                str(type_node.cols)
                if type_node.rows == type_node.cols
                else f"{type_node.cols}x{type_node.rows}"
            )
            prefix = {"double": "d"}.get(element_type, "")
            return f"{prefix}mat{dims}"
        if isinstance(type_node, ArrayType):
            # The outermost size comes first, as it was written: ``float[2][3]`` is
            # two arrays of three floats (see ``wrap_array_dimensions``).
            suffixes = []
            while isinstance(type_node, ArrayType):
                size = type_node.size
                if size is None:
                    suffixes.append("[]")
                elif isinstance(size, int):
                    suffixes.append(f"[{size}]")
                else:
                    suffixes.append(f"[{self.format_type_argument(size)}]")
                type_node = type_node.element_type
            return self.format_type_argument(type_node) + "".join(suffixes)
        if hasattr(type_node, "name"):
            if type_node.name is None:
                # An expression or node with no spelling here (a range, for one) names no type or value argument.
                raise CrossGLSyntaxError(
                    f"A {type(type_node).__name__} cannot be a type argument"
                )
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(self.format_type_argument(arg) for arg in generic_args)
                return f"{type_node.name}<{args}>"
            return type_node.name
        if hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.format_type_argument(type_node.element_type)
            if element_type == "float":
                return f"vec{type_node.size}"
            if element_type == "int":
                return f"ivec{type_node.size}"
            if element_type == "uint":
                return f"uvec{type_node.size}"
            if element_type == "double":
                return f"dvec{type_node.size}"
            if element_type == "bool":
                return f"bvec{type_node.size}"
            return f"{element_type}{type_node.size}"
        if isinstance(type_node, ASTNode):
            raise CrossGLSyntaxError(
                f"A {type(type_node).__name__} cannot be a type argument"
            )
        return str(type_node)

"""Parser that builds CrossGL AST nodes from lexer tokens."""

import logging

from .anonymous_blocks import bind_anonymous_buffer_block_members
from .ast import (
    ArrayType,
    AttributeNode,
    ConstantNode,
    EnumNode,
    ExecutionModel,
    FunctionNode,
    ImportNode,
    MatrixType,
    NamedType,
    PointerType,
    PreprocessorNode,
    PrimitiveType,
    ReferenceType,
    ShaderNode,
    StageMap,
    StageNode,
    StructNode,
    VariableNode,
    VectorType,
)
from .explicit_specializations import (
    EXPLICIT_SPECIALIZATION_ATTRIBUTE,
    explicit_specialization_annotation,
)
from .host_workgroup_size import resolve_host_workgroup_sizes
from .parsing.attributes import ParserAttributeMixin
from .parsing.cooperative_matrices import ParserCooperativeMatrixMixin
from .parsing.declaration_lookahead import ParserDeclarationLookaheadMixin
from .parsing.functions import ParserFunctionMixin
from .parsing.named_types import ParserNamedTypeMixin
from .parsing.operator_expressions import ParserOperatorExpressionMixin
from .parsing.postfix_expressions import ParserPostfixExpressionMixin
from .parsing.program_structure import ParserProgramStructureMixin
from .parsing.statements import ParserStatementMixin
from .parsing.type_arguments import ParserTypeArgumentMixin
from .parsing.type_declarations import ParserTypeDeclarationMixin
from .parsing.variable_declarations import ParserVariableDeclarationMixin
from .parsing.vocabulary import (
    SHADER_STAGE_TOKEN_TYPES,
    TEXTURE_TYPE_NAMES,
)
from .refusals import CrossGLSyntaxError
from .source_licenses import SOURCE_LICENSES
from .stage_utils import shader_stage_from_name
from .validation import validate_shader_cbuffers


class CrossGLFunctionBodyParseError(SyntaxError):
    """Raised when a function body cannot be represented by the shared IR."""

    project_diagnostic_code = "project.translate.crossgl-function-body-parse-failed"
    missing_capabilities = ("crossgl.function-body-parsing",)

    def __init__(self, function_name, reason, token):
        token_type, token_value = token
        rendered_token = token_type
        if token_value is not None:
            rendered_token = f"{token_type} {token_value!r}"
        super().__init__(
            f"CrossGL could not parse the body of function {function_name!r} "
            f"at {rendered_token}: {reason}"
        )
        self.function_name = function_name
        self.reason = str(reason)
        self.token_type = token_type
        self.token_value = token_value


class Parser(
    ParserProgramStructureMixin,
    ParserAttributeMixin,
    ParserTypeDeclarationMixin,
    ParserFunctionMixin,
    ParserVariableDeclarationMixin,
    ParserCooperativeMatrixMixin,
    ParserTypeArgumentMixin,
    ParserStatementMixin,
    ParserOperatorExpressionMixin,
    ParserPostfixExpressionMixin,
    ParserDeclarationLookaheadMixin,
    ParserNamedTypeMixin,
):
    """Recursive-descent parser for CrossGL Universal IR tokens."""

    def __init__(self, tokens):
        self.tokens = tokens
        # Every in-place token edit, as (position, replaced token), so that a
        # lookahead can undo its edits without copying the whole token list.
        self.token_edits = []
        self.anonymous_buffer_blocks = []
        # static_assert declarations outside blocks, attached to the ShaderNode by finalize_shader.
        self.static_assertions = []
        self.active_generic_parameter_names = set()
        self.value_type_scopes = [{}]
        # Declared values with their own parameters, which `name[...]` specializes.
        self.parameterized_value_names = set()
        self.current_function_return_type = None
        self.declared_type_names = {
            str(tokens[index + 1][1])
            for index, token in enumerate(tokens[:-1])
            if token[0] in {"STRUCT", "ENUM", "TRAIT"}
            and tokens[index + 1][0] == "IDENTIFIER"
        }
        self.pos = 0
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
        )

    def enter_generic_parameter_scope(self, generic_params):
        """Expose generic parameter names while parsing their declaration body."""
        previous = self.active_generic_parameter_names
        self.active_generic_parameter_names = previous | {
            parameter.name for parameter in generic_params
        }
        return previous

    def restore_generic_parameter_scope(self, previous):
        self.active_generic_parameter_names = previous

    def enter_value_type_scope(self, bindings=None):
        """Enter a lexical value-type scope with optional named bindings."""
        self.value_type_scopes.append(dict(bindings or {}))

    def restore_value_type_scope(self):
        """Leave the current lexical value-type scope."""
        self.value_type_scopes.pop()

    def register_value_type(self, name, value_type):
        """Register a declared value type in the current lexical scope."""
        if name:
            self.value_type_scopes[-1][str(name)] = value_type

    def lookup_value_type(self, name):
        """Resolve a value type from the innermost lexical scope."""
        for scope in reversed(self.value_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):
        """Consume one token of the expected type or raise ``SyntaxError``.

        The end of input is never consumed: a construct that tries to read past it is unterminated.
        """
        if token_type == "EOF" and self.current_token[0] == "EOF":
            raise CrossGLSyntaxError("Unexpected end of input")
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise CrossGLSyntaxError(
                f"Expected {token_type}, got {self.current_token[0]} '{self.current_token[1]}'"
            )

    def current_token_is_type_greater_than(self):
        """Return whether current token can close a generic type argument."""
        return self.current_token[0] in {"GREATER_THAN", "BITWISE_SHIFT_RIGHT"}

    def eat_type_greater_than(self):
        """Consume one generic ``>`` close, splitting lexer ``>>`` when needed."""
        if self.current_token[0] == "BITWISE_SHIFT_RIGHT":
            self.token_edits.append((self.pos, self.tokens[self.pos]))
            self.tokens[self.pos] = ("GREATER_THAN", ">")
            self.tokens.insert(self.pos + 1, ("GREATER_THAN", ">"))
            self.current_token = self.tokens[self.pos]
            self.eat("GREATER_THAN")
            return
        self.eat("GREATER_THAN")

    def mark_token_edits(self):
        """Return a mark that ``undo_token_edits`` can restore the tokens to."""
        return len(self.token_edits)

    def undo_token_edits(self, mark):
        """Undo, newest first, every ``>>`` split made since ``mark``."""
        while len(self.token_edits) > mark:
            position, token = self.token_edits.pop()
            del self.tokens[position + 1]
            self.tokens[position] = token

    def peek(self, offset=1):
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return ("EOF", None)

    def finalize_shader(self, shader):
        bind_anonymous_buffer_block_members(shader, self.anonymous_buffer_blocks)
        shader.static_assertions = list(self.static_assertions)
        return resolve_host_workgroup_sizes(
            validate_shader_cbuffers(shader), self.layout_argument_to_string
        )

    def append_parsed_nodes(self, collection, parsed):
        """Append one parsed node or extend a list of parsed nodes."""
        if not parsed:
            return
        if isinstance(parsed, list):
            collection.extend(node for node in parsed if node)
            return
        collection.append(parsed)

    def parse(self):
        """Parse a complete CrossGL translation unit into a ``ShaderNode``."""
        structs = []
        functions = []
        global_variables = []
        constants = []
        cbuffers = []
        stages = StageMap()
        imports = []
        preprocessors = []
        shader_name = "main"

        while self.current_token[0] != "EOF":
            previous_pos = self.pos

            parsed_element = self.parse_global()
            parsed_elements = (
                parsed_element if isinstance(parsed_element, list) else [parsed_element]
            )
            for parsed_element in parsed_elements:
                if not parsed_element:
                    continue
                if isinstance(parsed_element, ShaderNode):
                    shader_name = parsed_element.name or shader_name
                    structs.extend(parsed_element.structs)
                    functions.extend(parsed_element.functions)
                    global_variables.extend(parsed_element.global_variables)
                    constants.extend(parsed_element.constants)
                    imports.extend(parsed_element.imports)
                    preprocessors.extend(getattr(parsed_element, "preprocessors", []))
                    stages.update(parsed_element.stages)
                    cbuffers.extend(getattr(parsed_element, "cbuffers", []))
                elif isinstance(parsed_element, StructNode):
                    if getattr(parsed_element, "is_cbuffer", False):
                        cbuffers.append(parsed_element)
                    else:
                        structs.append(parsed_element)
                elif isinstance(parsed_element, FunctionNode):
                    functions.append(parsed_element)
                elif isinstance(parsed_element, VariableNode):
                    global_variables.append(parsed_element)
                elif isinstance(parsed_element, ConstantNode):
                    constants.append(parsed_element)
                elif isinstance(parsed_element, StageNode):
                    stages.append(parsed_element.stage, parsed_element)
                elif isinstance(parsed_element, ImportNode):
                    imports.append(parsed_element)
                elif isinstance(parsed_element, PreprocessorNode):
                    preprocessors.append(parsed_element)
                elif isinstance(parsed_element, EnumNode):
                    structs.append(parsed_element)

            if self.pos == previous_pos:
                self.reject_unexpected_token("a top-level declaration")

        shader = ShaderNode(
            name=shader_name,
            execution_model=ExecutionModel.GRAPHICS_PIPELINE,
            stages=stages,
            structs=structs,
            functions=functions,
            global_variables=global_variables,
            constants=constants,
            imports=imports,
            preprocessors=preprocessors,
        )
        if cbuffers:
            shader.cbuffers = cbuffers
        return self.finalize_shader(shader)

    def parse_function(self):
        """Parse a function declaration or definition."""
        qualifiers = []
        attributes = []

        while (
            self.current_token[0]
            in {
                "ASYNC",
                "UNSAFE",
                "GLOBAL",
                "KERNEL",
                "AT",
                "ATTRIBUTE",
            }
            or self.current_token_starts_square_attribute()
            or self.current_token_is_function_precise_qualifier()
        ):
            if self.current_token[0] in ["AT", "ATTRIBUTE"]:
                attributes.extend(self.parse_attributes())
            elif self.current_token_starts_square_attribute():
                attributes.extend(self.parse_square_attributes())
            elif self.current_token_is_function_precise_qualifier():
                # Kept in the attribute form that codegen honors for functions;
                # leading qualifiers name stages for several targets.
                self.eat("IDENTIFIER")
                if not any(
                    getattr(attribute, "name", None) == "precise"
                    for attribute in attributes
                ):
                    attributes.append(AttributeNode(name="precise"))
            else:
                qualifiers.append(self.current_token[1])
                self.eat(self.current_token[0])

        saw_function_keyword = False
        if self.current_token[0] == "FUNCTION":
            self.eat("FUNCTION")
            saw_function_keyword = True

        if saw_function_keyword and self.current_token_starts_bare_function_name():
            return_type = PrimitiveType("void")
        else:
            return_type = self.parse_function_return_type()

        if not (
            self.current_token[0] == "KERNEL"
            or self.current_token_is_binding_identifier()
        ):
            raise CrossGLSyntaxError(
                f"Expected function name, got {self.current_token[0]}"
            )
        name = self.parse_binding_identifier()

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        previous_scope = self.enter_generic_parameter_scope(generic_params)
        try:
            self.eat("LPAREN")
            parameters = self.parse_parameter_list()
            self.eat("RPAREN")

            if self.is_arrow_token():
                self.eat_arrow()
                attributes.extend(self.parse_return_type_attributes())
                return_type = self.parse_function_return_type()

            post_attributes = self.parse_post_declaration_attributes()

            body = None
            if self.current_token[0] == "LBRACE":
                parameter_types = {
                    parameter.name: parameter.param_type for parameter in parameters
                }
                previous_return_type = self.current_function_return_type
                self.current_function_return_type = return_type
                self.enter_value_type_scope(parameter_types)
                try:
                    try:
                        body = self.parse_block()
                    except SyntaxError as exc:
                        # A body that does not parse is refused: an empty body in its place would be a function
                        # that returns an undefined value.
                        raise CrossGLFunctionBodyParseError(
                            name,
                            exc,
                            self.current_token,
                        ) from exc
                finally:
                    self.restore_value_type_scope()
                    self.current_function_return_type = previous_return_type
            else:
                self.eat("SEMICOLON")
        finally:
            self.restore_generic_parameter_scope(previous_scope)

        function_attributes = []
        linkage_qualifiers = set()
        source_licenses = set()
        explicit_specialization = None
        for attribute in attributes + post_attributes:
            if attribute.name == "source_license":
                arguments = attribute.arguments
                license_name = (
                    getattr(arguments[0], "name", None) if len(arguments) == 1 else None
                )
                if license_name not in SOURCE_LICENSES:
                    raise CrossGLSyntaxError(
                        "@source_license requires one registered license identifier"
                    )
                source_licenses.add(license_name)
            elif attribute.name == EXPLICIT_SPECIALIZATION_ATTRIBUTE:
                explicit_specialization = explicit_specialization_annotation(attribute)
            elif attribute.name in {"metal_static", "metal_inline"}:
                if attribute.arguments:
                    raise CrossGLSyntaxError(
                        f"@{attribute.name} does not accept arguments"
                    )
                qualifier = attribute.name[len("metal_") :]
                linkage_qualifiers.add(qualifier)
            else:
                function_attributes.append(attribute)

        return FunctionNode(
            name=name,
            return_type=return_type,
            parameters=parameters,
            body=body,
            generic_params=generic_params,
            attributes=function_attributes,
            qualifiers=qualifiers,
            is_async="async" in qualifiers,
            is_unsafe="unsafe" in qualifiers,
            linkage="internal" if "static" in linkage_qualifiers else "external",
            is_inline="inline" in linkage_qualifiers,
            annotations=(
                {
                    **(
                        {"source_licenses": tuple(sorted(source_licenses))}
                        if source_licenses
                        else {}
                    ),
                    **(
                        {EXPLICIT_SPECIALIZATION_ATTRIBUTE: explicit_specialization}
                        if explicit_specialization
                        else {}
                    ),
                }
                or None
            ),
        )

    def parse_type(self):
        """Parse a CrossGL type expression into a ``TypeNode``."""
        is_buffer = False
        if self.current_token[0] == "BUFFER":
            is_buffer = True
            self.eat("BUFFER")

        base_type = None

        if self.current_token[0] in [
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
        ]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
            base_type = PrimitiveType(type_name)

        elif self.current_token[0] in [
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
        ]:
            vec_type = self.current_token[1]
            self.eat(self.current_token[0])
            generic_args = []
            if self.current_token[0] == "LESS_THAN":
                generic_args = self.parse_generic_arguments()

            if generic_args:
                element_type = self.vector_element_type_from_generic(generic_args[0])
                size = int(vec_type[-1])
            elif vec_type.startswith("ivec"):
                element_type = PrimitiveType("int")
                size = int(vec_type[-1])
            elif vec_type.startswith("uvec"):
                element_type = PrimitiveType("uint")
                size = int(vec_type[-1])
            elif vec_type.startswith("dvec"):
                element_type = PrimitiveType("double")
                size = int(vec_type[-1])
            elif vec_type.startswith("bvec"):
                element_type = PrimitiveType("bool")
                size = int(vec_type[-1])
            else:  # vec
                element_type = PrimitiveType("float")
                size = int(vec_type[-1])

            base_type = VectorType(element_type, size)

        elif self.current_token[0] in [
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
        ]:
            mat_type = self.current_token[1]
            self.eat(self.current_token[0])

            is_double_matrix = mat_type.startswith("dmat")
            dimensions = mat_type[4:] if is_double_matrix else mat_type[3:]
            if "x" in dimensions:
                # CrossGL follows GLSL: matCxR has C columns of R-component vectors.
                cols, rows = map(int, dimensions.split("x"))
            else:
                size = int(dimensions)
                rows = cols = size

            element_type = "double" if is_double_matrix else "float"
            base_type = MatrixType(PrimitiveType(element_type), rows, cols)

        elif self.current_token[0] in [
            "TEXTURE1D",
            "TEXTURE2D",
            "TEXTURE3D",
            "TEXTURECUBE",
            "TEXTURE2DARRAY",
        ]:
            token_type = self.current_token[0]
            type_name = TEXTURE_TYPE_NAMES[token_type]
            self.eat(token_type)

            generic_args = []
            if self.current_token[0] == "LESS_THAN":
                generic_args = self.parse_generic_arguments()

            base_type = NamedType(type_name, generic_args)

        elif self.current_token[0] in [
            "SAMPLER",
            "SAMPLER1D",
            "SAMPLER1DARRAY",
            "SAMPLER2D",
            "SAMPLER3D",
            "SAMPLERCUBE",
            "SAMPLER2DARRAY",
            "SAMPLER2DSHADOW",
            "SAMPLER2DARRAYSHADOW",
            "SAMPLERCUBESHADOW",
            "SAMPLERCUBEARRAY",
            "SAMPLERCUBEARRAYSHADOW",
            "SAMPLER2DMS",
            "SAMPLER2DMSARRAY",
            "ISAMPLER1D",
            "ISAMPLER1DARRAY",
            "ISAMPLER2D",
            "ISAMPLER3D",
            "ISAMPLERCUBE",
            "ISAMPLERCUBEARRAY",
            "ISAMPLER2DARRAY",
            "ISAMPLER2DMS",
            "ISAMPLER2DMSARRAY",
            "USAMPLER1D",
            "USAMPLER1DARRAY",
            "USAMPLER2D",
            "USAMPLER3D",
            "USAMPLERCUBE",
            "USAMPLERCUBEARRAY",
            "USAMPLER2DARRAY",
            "USAMPLER2DMS",
            "USAMPLER2DMSARRAY",
            "IIMAGE1D",
            "IIMAGE1DARRAY",
            "IIMAGE2D",
            "IIMAGE3D",
            "IIMAGECUBE",
            "IIMAGECUBEARRAY",
            "IIMAGE2DARRAY",
            "IIMAGE2DMS",
            "IIMAGE2DMSARRAY",
            "UIMAGE1D",
            "UIMAGE1DARRAY",
            "UIMAGE2D",
            "UIMAGE3D",
            "UIMAGECUBE",
            "UIMAGECUBEARRAY",
            "UIMAGE2DARRAY",
            "UIMAGE2DMS",
            "UIMAGE2DMSARRAY",
            "IMAGE1D",
            "IMAGE1DARRAY",
            "IMAGE2D",
            "IMAGE3D",
            "IMAGECUBE",
            "IMAGECUBEARRAY",
            "IMAGE2DARRAY",
            "IMAGE2DMS",
            "IMAGE2DMSARRAY",
        ]:
            base_type = self.parse_opaque_resource_type()

        elif self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "def":
            base_type = self.parse_callable_type()

        elif (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "array"
            and self.peek()[0] == "LESS_THAN"
        ):
            base_type = self.parse_array_template_type()

        elif self.current_token[0] == "IDENTIFIER":
            base_type = self.parse_named_base_type()

        elif self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            element_type = self.parse_type()

            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()

            self.eat("RBRACKET")
            base_type = ArrayType(element_type, size)

        else:
            base_type = PrimitiveType("float")

        sizes = []
        while self.current_token[0] == "LBRACKET":
            if (
                not sizes
                and isinstance(base_type, NamedType)
                and self.square_brackets_form_type_arguments()
            ):
                base_type.generic_args.extend(self.parse_square_generic_arguments())
                continue
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            sizes.append(size)
        base_type = self.wrap_array_dimensions(base_type, sizes)

        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            if self.current_token[0] == "MULTIPLY":
                self.eat("MULTIPLY")
                base_type = PointerType(base_type)
                continue

            self.eat("BITWISE_AND")
            is_mutable_reference = False
            if self.current_token[0] == "MUT":
                self.eat("MUT")
                is_mutable_reference = True
            base_type = ReferenceType(base_type, is_mutable=is_mutable_reference)

        if is_buffer:
            if hasattr(base_type, "qualifiers"):
                base_type.qualifiers = getattr(base_type, "qualifiers", []) + ["buffer"]
            else:
                base_type = NamedType(f"buffer_{base_type}", [])

        return base_type

    # Operand starts that unambiguously begin a value (a "primary"). Used for
    # identifier-headed casts like ``(int64)tid`` where the type name is lexed
    # as an identifier: only these keep ``(name)operand`` distinct from a
    # parenthesized variable followed by a binary operator (``(a) + b``).
    UNAMBIGUOUS_CAST_OPERAND_START_TOKENS = frozenset(
        {
            "IDENTIFIER",
            "NUMBER",
            "FLOAT_NUMBER",
            "HEX_NUMBER",
            "BIN_NUMBER",
            "OCT_NUMBER",
            "BOOLEAN_LITERAL",
            "STRING_LITERAL",
            "CHAR_LITERAL",
            "LPAREN",
        }
    )
    # Additional operand starts allowed after a *keyword* type cast, where the
    # parenthesized token is unambiguously a type (``(int)-x`` is a cast).
    CAST_OPERAND_START_TOKENS = UNAMBIGUOUS_CAST_OPERAND_START_TOKENS | {
        "NOT",
        "MINUS",
        "PLUS",
        "MULTIPLY",
        "BITWISE_NOT",
        "BITWISE_AND",
        "INCREMENT",
        "DECREMENT",
    }

    # Keywords that continue an expression (``(a) as T``, ``x in r``) rather
    # than start an operand.
    EXPRESSION_CONTINUATION_KEYWORDS = frozenset({"AS", "IN", "WHERE", "MATCH"})

    def create_empty_shader(self):
        """Create an empty shader used by parser error recovery."""
        return ShaderNode(
            name="error",
            execution_model=ExecutionModel.GRAPHICS_PIPELINE,
            stages={},
            structs=[],
            functions=[],
            global_variables=[],
            constants=[],
            imports=[],
        )

    def report_error(self, message):
        """Emit a parser warning."""
        logging.warning(f"Parser: {message}")

    def next_token(self):
        """Advance to the next token without validating token type."""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]

    def current_token_starts_shader_stage_block(self):
        """Return whether the current token starts an explicit stage block."""
        if self.current_token[0] in SHADER_STAGE_TOKEN_TYPES:
            return True

        if shader_stage_from_name(self.current_token[1]) is None:
            return False

        return self.peek()[0] == "LBRACE" or (
            self.peek()[0] == "IDENTIFIER" and self.peek(2)[0] == "LBRACE"
        )

    def parse_global(self):
        """Parse one top-level declaration; input that starts none raises ``SyntaxError``."""
        if self.current_token[0] == "EOF":
            return None

        if self.parse_scope_static_assertion():
            return None

        if self.current_token[0] == "PREPROCESSOR":
            return self.parse_preprocessor_directive()

        if self.current_token[0] == "PRECISION":
            return self.parse_precision_statement()

        if self.current_token[0] in ["IMPORT", "USE", "FROM"]:
            return self.parse_import()

        if self.current_token[0] == "SHADER":
            return self.parse_shader_declaration()

        if self.is_generic_declaration():
            return self.parse_generic_declaration()

        if self.current_token[0] == "TRAIT":
            return self.parse_trait()

        if self.current_token[0] == "ENUM":
            return self.parse_enum()

        if self.current_token_starts_attributed_cbuffer_declaration():
            return self.parse_attributed_cbuffer()

        if self.current_token_starts_attributed_struct_declaration():
            return self.parse_attributed_struct()

        if self.current_token[0] == "STRUCT":
            return self.parse_struct()

        if self.current_token[0] == "CONST" and not self.is_function_declaration():
            return self.parse_constant()

        if self.current_token[0] == "LET":
            return self.parse_let_declaration()

        if self.is_cbuffer_declaration():
            return self.parse_cbuffer_as_struct()

        if self.current_token[0] == "LAYOUT":
            attributes = self.parse_layout_attributes()
            if self.is_layout_buffer_block_declaration():
                return list(self.parse_layout_buffer_block(attributes))
            if self.current_token_starts_attributed_cbuffer_declaration():
                return self.parse_attributed_cbuffer(attributes)
            if self.is_cbuffer_declaration():
                return self.parse_cbuffer_as_struct(attributes)
            if self.is_variable_declaration():
                return self.parse_variable_declaration(attributes)
            self.reject_unexpected_token("a layout-qualified declaration")

        if self.current_token_starts_shader_stage_block():
            return self.parse_shader_stage_block()

        # Functions take priority over variables to avoid misidentification.
        if self.is_function_declaration():
            return self.parse_function()

        if self.is_variable_declaration():
            return self.parse_variable_declaration()

        # is_function_declaration can fail on some edge-case type tokens; try anyway.
        if self.is_type_token():
            return self.parse_function()

        if self.current_token[0] == "SEMICOLON":
            # An empty declaration, which C++ [dcl.pre] and GLSL allow at namespace or global scope.
            self.eat("SEMICOLON")
            return None
        self.reject_unexpected_token("a top-level declaration")

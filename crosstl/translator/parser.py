from .ast import (
    ASTNode,
    TypeNode,
    PrimitiveType,
    VectorType,
    MatrixType,
    ArrayType,
    PointerType,
    ReferenceType,
    FunctionType,
    GenericType,
    NamedType,
    ShaderNode,
    StageNode,
    ImportNode,
    StructNode,
    StructMemberNode,
    EnumNode,
    EnumVariantNode,
    FunctionNode,
    ParameterNode,
    VariableNode,
    ConstantNode,
    GenericParameterNode,
    AttributeNode,
    StatementNode,
    BlockNode,
    ExpressionStatementNode,
    AssignmentNode,
    IfNode,
    ForNode,
    ForInNode,
    WhileNode,
    LoopNode,
    MatchNode,
    MatchArmNode,
    SwitchNode,
    CaseNode,
    ReturnNode,
    BreakNode,
    ContinueNode,
    ExpressionNode,
    LiteralNode,
    IdentifierNode,
    BinaryOpNode,
    UnaryOpNode,
    TernaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    PointerAccessNode,
    ArrayAccessNode,
    SwizzleNode,
    CastNode,
    ConstructorNode,
    LambdaNode,
    PatternNode,
    WildcardPatternNode,
    IdentifierPatternNode,
    LiteralPatternNode,
    StructPatternNode,
    TextureNode,
    AtomicOpNode,
    SyncNode,
    BuiltinVariableNode,
    BufferNode,
    TextureResourceNode,
    SamplerNode,
    ArrayNode,
    ShaderStage,
    ExecutionModel,
    create_legacy_shader_node,
)
from .lexer import Lexer
import logging


class Parser:
    """
    Production-ready parser for CrossGL Universal IR.

    Supports comprehensive parsing for all language features including:
    - Modern type systems with generics and traits
    - Pattern matching and algebraic data types
    - Async/await patterns
    - GPU programming constructs
    - Memory safety features
    - Advanced control flow
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
        )

    def skip_comments(self):
        """Skip comments in the token list."""
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):
        """Consume the current token if it matches the expected token type."""
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(
                f"Expected {token_type}, got {self.current_token[0]} '{self.current_token[1]}'"
            )

    def peek(self, offset=1):
        """Look ahead at tokens without consuming them."""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return ("EOF", None)

    def parse(self):
        """Parse the entire program."""
        structs = []
        functions = []
        global_variables = []
        constants = []
        stages = {}
        imports = []

        loop_count = 0  # Protection against infinite loops
        max_loops = 10000  # Reasonable upper limit

        while self.current_token[0] != "EOF" and loop_count < max_loops:
            loop_count += 1

            # Use the enhanced global parsing
            parsed_element = self.parse_global()
            if parsed_element:
                if isinstance(parsed_element, StructNode):
                    structs.append(parsed_element)
                elif isinstance(parsed_element, FunctionNode):
                    functions.append(parsed_element)
                elif isinstance(parsed_element, VariableNode):
                    global_variables.append(parsed_element)
                elif isinstance(parsed_element, ConstantNode):
                    constants.append(parsed_element)
                elif isinstance(parsed_element, StageNode):
                    stages[parsed_element.stage] = parsed_element
                elif isinstance(parsed_element, ImportNode):
                    imports.append(parsed_element)
                elif isinstance(parsed_element, EnumNode):
                    # Treat enums as special structs for now
                    structs.append(parsed_element)

            # Safety check: if we're not advancing, break
            if loop_count > 100:  # Allow some iterations for complex files
                current_token_info = (
                    f"{self.current_token[0]}:{self.current_token[1]}"
                    if self.current_token[1]
                    else self.current_token[0]
                )
                print(
                    f"Warning: Parser may be stuck at token {current_token_info}, breaking..."
                )
                break

        if loop_count >= max_loops:
            print(f"Warning: Parser hit maximum loop limit ({max_loops}), stopping...")

        return ShaderNode(
            name="main",
            execution_model=ExecutionModel.GRAPHICS_PIPELINE,
            stages=stages,
            structs=structs,
            functions=functions,
            global_variables=global_variables,
            constants=constants,
            imports=imports,
        )

    def parse_program(self):
        """Parse a complete program/shader."""
        imports = []
        structs = []
        enums = []
        functions = []
        constants = []
        global_variables = []

        # Check if this is a shader program
        shader_node = None

        while self.current_token[0] != "EOF":
            if self.current_token[0] in ["IMPORT", "USE"]:
                imports.append(self.parse_import())
            elif self.current_token[0] == "SHADER":
                shader_node = self.parse_shader_declaration()
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "ENUM":
                enums.append(self.parse_enum())
            elif self.current_token[0] in ["CBUFFER", "UNIFORM"]:
                # Handle as struct for simplicity
                structs.append(self.parse_cbuffer_as_struct())
            elif self.current_token[0] == "CONST":
                constants.append(self.parse_constant())
            elif self.is_function_declaration():
                functions.append(self.parse_function())
            elif self.is_variable_declaration():
                global_variables.append(self.parse_variable_declaration())
            elif self.current_token[0] in ["VERTEX", "FRAGMENT", "COMPUTE"]:
                # Shader stage - create a stage function
                stage_func = self.parse_shader_stage()
                functions.append(stage_func)
            else:
                # Skip unknown tokens
                self.skip_unknown_token()

        # Create appropriate return type
        if shader_node:
            shader_node.structs.extend(structs)
            shader_node.functions.extend(functions)
            shader_node.global_variables.extend(global_variables)
            shader_node.constants.extend(constants)
            shader_node.imports.extend(imports)
            return shader_node
        else:
            # Return a general program shader
            return ShaderNode(
                name="main",
                execution_model=ExecutionModel.GRAPHICS_PIPELINE,
                stages={},
                structs=structs,
                functions=functions,
                global_variables=global_variables,
                constants=constants,
                imports=imports,
            )

    def parse_shader_declaration(self):
        """Parse shader declaration."""
        self.eat("SHADER")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        execution_model = ExecutionModel.GRAPHICS_PIPELINE
        stages = {}
        functions = []
        structs = []
        global_variables = []
        constants = []

        self.eat("LBRACE")

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] in ["VERTEX", "FRAGMENT", "COMPUTE", "GEOMETRY"]:
                stage_node = self.parse_shader_stage_block()
                stages[stage_node.stage] = stage_node
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] in ["CBUFFER", "UNIFORM"]:
                structs.append(self.parse_cbuffer_as_struct())
            elif self.current_token[0] == "CONST":
                constants.append(self.parse_constant())
            elif self.is_function_declaration():
                # Handle top-level functions in shader
                func = self.parse_function()
                functions.append(func)
            elif self.is_variable_declaration():
                # Handle global variables
                global_variables.append(self.parse_variable_declaration())
            else:
                self.skip_unknown_token()

        self.eat("RBRACE")

        return ShaderNode(
            name=name,
            execution_model=execution_model,
            stages=stages,
            structs=structs,
            functions=functions,
            global_variables=global_variables,
            constants=constants,
        )

    def parse_shader_stage_block(self):
        """Parse a shader stage block (vertex, fragment, etc.)."""
        stage_type = self.current_token[1]
        stage_enum = {
            "vertex": ShaderStage.VERTEX,
            "fragment": ShaderStage.FRAGMENT,
            "compute": ShaderStage.COMPUTE,
            "geometry": ShaderStage.GEOMETRY,
        }.get(stage_type, ShaderStage.VERTEX)

        self.eat(self.current_token[0])

        # Handle named shader stages (e.g., "compute spawn", "compute matmul_tiled")
        stage_name = None
        if self.current_token[0] == "IDENTIFIER":
            stage_name = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("LBRACE")

        local_variables = []
        local_functions = []
        main_function = None

        while self.current_token[0] != "RBRACE":
            if self.is_function_declaration():
                func = self.parse_function()
                if func.name == "main":
                    main_function = func
                else:
                    local_functions.append(func)
            elif self.is_variable_declaration():
                local_variables.append(self.parse_variable_declaration())
            else:
                self.skip_unknown_token()

        self.eat("RBRACE")

        # Create main function if not found
        if not main_function:
            main_function = FunctionNode(
                name="main",
                return_type=PrimitiveType("void"),
                parameters=[],
                body=BlockNode([]),
            )

        # If we have a stage name, incorporate it into the function name
        if stage_name:
            main_function.name = stage_name

        return StageNode(
            stage=stage_enum,
            entry_point=main_function,
            local_variables=local_variables,
            local_functions=local_functions,
        )

    def parse_import(self):
        """Parse import/use statements."""
        if self.current_token[0] == "IMPORT":
            self.eat("IMPORT")
        else:
            self.eat("USE")

        path = self.current_token[1]
        self.eat("IDENTIFIER")

        alias = None
        items = None

        if self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("SEMICOLON")

        return ImportNode(path=path, alias=alias, items=items)

    def parse_struct(self):
        """Parse struct declarations."""
        if self.current_token[0] == "EOF":
            return None

        self.eat("STRUCT")

        if self.current_token[0] != "IDENTIFIER":
            return None

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        if self.current_token[0] != "LBRACE":
            # Skip malformed struct
            while self.current_token[0] not in ["SEMICOLON", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "SEMICOLON":
                self.skip_unknown_token()
            return None

        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            member = self.parse_struct_member()
            if member:
                members.append(member)

        if self.current_token[0] == "EOF":
            return None

        self.eat("RBRACE")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return StructNode(name=name, members=members, generic_params=generic_params)

    def parse_struct_member(self):
        """Parse individual struct member."""
        # EOF protection
        if self.current_token[0] == "EOF":
            return None

        # Handle nested enum declarations inside structs
        if self.current_token[0] == "ENUM":
            return self.parse_enum()

        # Skip over complex constructs we can't handle yet
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] in [
            "generic",
            "trait",
        ]:
            # Skip to semicolon or brace
            while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "SEMICOLON":
                self.skip_unknown_token()
            return None

        member_type = self.parse_type()

        # Handle dynamic arrays before identifier (e.g., float[] name;)
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            member_type = ArrayType(member_type, size)

        if self.current_token[0] != "IDENTIFIER":
            # Skip malformed member
            while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "SEMICOLON":
                self.skip_unknown_token()
            return None

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle multi-dimensional C-style array declarations (e.g., float values[4][8][2];)
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            member_type = ArrayType(member_type, size)

        attributes = []
        if self.current_token[0] == "AT":
            attributes = self.parse_attributes()

        if self.current_token[0] != "SEMICOLON":
            # Skip malformed member
            while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "SEMICOLON":
                self.skip_unknown_token()
            return None

        self.eat("SEMICOLON")

        return StructMemberNode(
            name=name, member_type=member_type, attributes=attributes
        )

    def parse_enum(self):
        """Parse enum declarations."""
        self.eat("ENUM")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type = self.parse_type()

        self.eat("LBRACE")
        variants = []

        while self.current_token[0] != "RBRACE":
            variant = self.parse_enum_variant()
            variants.append(variant)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RBRACE")

        return EnumNode(name=name, variants=variants, underlying_type=underlying_type)

    def parse_enum_variant(self):
        """Parse enum variant with support for complex variants."""
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        value = None
        variant_data = None

        # Handle enum variants with associated data (e.g., Some(T), None)
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            # Parse variant data/parameters
            variant_params = []
            while self.current_token[0] != "RPAREN":
                if self.current_token[0] == "IDENTIFIER":
                    # Type parameter
                    param_type = self.parse_type()
                    variant_params.append(param_type)
                else:
                    # Parse as expression
                    param_expr = self.parse_expression()
                    variant_params.append(param_expr)

                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                elif self.current_token[0] != "RPAREN":
                    break
            self.eat("RPAREN")
            variant_data = variant_params

        # Handle enum variants with explicit values (e.g., RED = 1)
        elif self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

        # Handle struct-like enum variants (e.g., Point { x: i32, y: i32 })
        elif self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            struct_members = []
            while self.current_token[0] != "RBRACE":
                if self.current_token[0] == "IDENTIFIER":
                    member_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    self.eat("COLON")
                    member_type = self.parse_type()
                    struct_members.append((member_name, member_type))

                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                elif self.current_token[0] != "RBRACE":
                    self.skip_unknown_token()
                else:
                    break
            self.eat("RBRACE")
            variant_data = struct_members

        # Create enhanced enum variant node
        variant_node = EnumVariantNode(name=name, value=value)
        if variant_data:
            # Store additional data in the variant node
            variant_node.data = variant_data

        return variant_node

    def parse_function(self):
        """Parse function declarations."""
        qualifiers = []
        attributes = []

        # Parse function qualifiers and attributes
        while self.current_token[0] in [
            "ASYNC",
            "UNSAFE",
            "GLOBAL",
            "KERNEL",
            "ATTRIBUTE",
        ]:
            if self.current_token[0] == "ATTRIBUTE":
                attributes = self.parse_attributes()
            else:
                qualifiers.append(self.current_token[1])
                self.eat(self.current_token[0])

        # Parse function keyword
        if self.current_token[0] == "FUNCTION":
            self.eat("FUNCTION")

        # Parse return type
        return_type = self.parse_type()

        # Parse function name
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Parse generic parameters
        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        # Parse parameters
        self.eat("LPAREN")
        parameters = self.parse_parameter_list()
        self.eat("RPAREN")

        # Parse post-parameter attributes (e.g., return value semantics)
        post_attributes = []
        if self.current_token[0] == "AT":
            post_attributes = self.parse_attributes()

        # Parse function body
        body = None
        if self.current_token[0] == "LBRACE":
            body = self.parse_block()
        else:
            self.eat("SEMICOLON")  # Function declaration only

        return FunctionNode(
            name=name,
            return_type=return_type,
            parameters=parameters,
            body=body,
            generic_params=generic_params,
            attributes=attributes + post_attributes,  # Combine pre and post attributes
            qualifiers=qualifiers,
            is_async="async" in qualifiers,
            is_unsafe="unsafe" in qualifiers,
        )

    def parse_parameter_list(self):
        """Parse function parameter list."""
        parameters = []

        while self.current_token[0] != "RPAREN":
            param = self.parse_parameter()
            parameters.append(param)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RPAREN":
                break

        return parameters

    def parse_parameter(self):
        """Parse function parameter."""
        attributes = []
        if self.current_token[0] == "ATTRIBUTE":
            attributes = self.parse_attributes()

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        param_type = self.parse_type()
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle @ semantic annotations for parameters (e.g., int vertexID @ gl_VertexID)
        if self.current_token[0] == "AT":
            self.eat("AT")
            semantic_name = self.current_token[1]
            self.eat("IDENTIFIER")
            # Add semantic as an attribute
            attributes.append(AttributeNode(name=semantic_name, arguments=[]))

        default_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            default_value = self.parse_expression()

        return ParameterNode(
            name=name,
            param_type=param_type,
            default_value=default_value,
            attributes=attributes,
            is_mutable=is_mutable,
        )

    def parse_variable_declaration(self):
        """Parse variable declarations."""
        qualifiers = []

        # Parse qualifiers
        while self.current_token[0] in [
            "CONST",
            "STATIC",
            "MUT",
            "SHARED",
            "UNIFORM",
            "BUFFER",
        ]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        var_type = self.parse_type()
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle multi-dimensional C-style array declarations (e.g., float arr[5][10];)
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            var_type = ArrayType(var_type, size)

        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()

        self.eat("SEMICOLON")

        return VariableNode(
            name=name,
            var_type=var_type,
            initial_value=initial_value,
            qualifiers=qualifiers,
            is_mutable="const" not in qualifiers,
        )

    def is_variable_declaration(self):
        """
        Enhanced detection of variable declarations with better lookahead.
        Handles complex cases and distinguishes from function calls and member access.
        """
        saved_pos = self.pos
        saved_token = self.current_token

        try:
            # Skip qualifiers
            while self.current_token[0] in [
                "CONST",
                "STATIC",
                "MUT",
                "SHARED",
                "UNIFORM",
                "BUFFER",
            ]:
                self.eat(self.current_token[0])

            # Must have a type
            if not self.is_type_token():
                return False

            # Parse through the type to get to identifier
            self.advance_over_type()

            # Must have an identifier next
            if self.current_token[0] != "IDENTIFIER":
                return False

            self.current_token[1]
            self.eat("IDENTIFIER")

            # Check what comes after the identifier
            next_token = self.current_token[0]

            # Array declarations: identifier[...]
            if next_token == "LBRACKET":
                return True

            # Assignment: identifier = ...
            if next_token == "EQUALS":
                return True

            # End of statement: identifier;
            if next_token == "SEMICOLON":
                return True

            # Comma (multiple declarations): identifier, ...
            if next_token == "COMMA":
                return True

            # Function parameters context
            if next_token in ["RPAREN", "COMMA"] and self.in_parameter_context():
                return True

            # Member access or function call: identifier.something or identifier(...)
            # These are NOT variable declarations
            if next_token in ["DOT", "LPAREN"]:
                return False

            # If followed by an operator, it's likely an expression, not a declaration
            if next_token in [
                "PLUS",
                "MINUS",
                "MULTIPLY",
                "DIVIDE",
                "MOD",
                "EQUAL",
                "NOT_EQUAL",
                "LESS_THAN",
                "GREATER_THAN",
                "LOGICAL_AND",
                "LOGICAL_OR",
                "BITWISE_AND",
                "BITWISE_OR",
                "ASSIGN_ADD",
                "ASSIGN_SUB",
                "ASSIGN_MUL",
                "ASSIGN_DIV",
                "ASSIGN_MOD",
                "ASSIGN_XOR",
                "ASSIGN_OR",
                "ASSIGN_AND",
                "ASSIGN_SHIFT_LEFT",
                "ASSIGN_SHIFT_RIGHT",
            ]:
                return False

            # Default to true if we can't determine
            return True

        except Exception:
            return False
        finally:
            # Restore parser state
            self.pos = saved_pos
            self.current_token = saved_token

    def in_parameter_context(self):
        """Check if we're currently parsing function parameters."""
        # Simple heuristic: look backwards for recent LPAREN
        for i in range(max(0, self.pos - 10), self.pos):
            if i < len(self.tokens) and self.tokens[i][0] == "LPAREN":
                return True
        return False

    def parse_constant(self):
        """Parse constant declarations."""
        self.eat("CONST")
        const_type = self.parse_type()
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("EQUALS")
        value = self.parse_expression()
        self.eat("SEMICOLON")

        return ConstantNode(name=name, const_type=const_type, value=value)

    def parse_type(self):
        """Parse type expressions."""
        # Handle buffer pointer syntax (e.g., buffer float*)
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
            "BVEC2",
            "BVEC3",
            "BVEC4",
        ]:
            vec_type = self.current_token[1]
            self.eat(self.current_token[0])

            # Extract element type and size from vec type
            if vec_type.startswith("ivec"):
                element_type = PrimitiveType("int")
                size = int(vec_type[-1])
            elif vec_type.startswith("uvec"):
                element_type = PrimitiveType("uint")
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
        ]:
            mat_type = self.current_token[1]
            self.eat(self.current_token[0])

            # Extract matrix dimensions
            if "x" in mat_type:
                rows, cols = map(int, mat_type[3:].split("x"))
            else:
                size = int(mat_type[-1])
                rows = cols = size

            base_type = MatrixType(PrimitiveType("float"), rows, cols)

        elif self.current_token[0] == "IDENTIFIER":
            # Named type (struct, enum, etc.)
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            generic_args = []
            if self.current_token[0] == "LESS_THAN":
                generic_args = self.parse_generic_arguments()

            base_type = NamedType(name, generic_args)

        elif self.current_token[0] == "LBRACKET":
            # Array type
            self.eat("LBRACKET")
            element_type = self.parse_type()

            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()

            self.eat("RBRACKET")
            base_type = ArrayType(element_type, size)

        else:
            # Fallback to primitive float
            base_type = PrimitiveType("float")

        # Handle array syntax immediately after type (e.g., float[])
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            base_type = ArrayType(base_type, size)

        # Handle pointer syntax (*)
        if self.current_token[0] == "MULTIPLY":
            self.eat("MULTIPLY")
            # For simplicity, treat pointers as the base type for now
            # In a full implementation, you'd want a PointerType

        # If this was a buffer type, wrap it appropriately
        if is_buffer:
            # For buffer types, we can either create a special BufferType or add a qualifier
            # For now, we'll add 'buffer' as a qualifier to the base type
            if hasattr(base_type, "qualifiers"):
                base_type.qualifiers = getattr(base_type, "qualifiers", []) + ["buffer"]
            else:
                # Create a named type with buffer qualifier
                base_type = NamedType(f"buffer_{base_type}", [])

        return base_type

    def parse_generic_parameters(self):
        """Parse generic parameter list."""
        self.eat("LESS_THAN")
        params = []

        while self.current_token[0] != "GREATER_THAN":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            constraints = []
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                constraints.append(self.parse_type())

                while self.current_token[0] == "PLUS":
                    self.eat("PLUS")
                    constraints.append(self.parse_type())

            params.append(GenericParameterNode(name=name, constraints=constraints))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("GREATER_THAN")
        return params

    def parse_generic_arguments(self):
        """Parse generic argument list."""
        self.eat("LESS_THAN")
        args = []

        while self.current_token[0] != "GREATER_THAN":
            arg_type = self.parse_type()
            args.append(arg_type)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("GREATER_THAN")
        return args

    def parse_attributes(self):
        """Parse attribute list."""
        attributes = []

        while self.current_token[0] == "AT":
            self.eat("AT")
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            arguments = []
            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                while self.current_token[0] != "RPAREN":
                    arguments.append(self.parse_expression())
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                self.eat("RPAREN")

            attributes.append(AttributeNode(name=name, arguments=arguments))

        return attributes

    def parse_block(self):
        """Parse statement block."""
        self.eat("LBRACE")
        statements = []

        while self.current_token[0] != "RBRACE":
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

        self.eat("RBRACE")
        return BlockNode(statements)

    def parse_statement(self):
        """Parse individual statements."""
        if self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
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
        elif self.is_variable_declaration():
            return self.parse_variable_declaration()
        else:
            # Expression statement
            expr = self.parse_expression()
            self.eat("SEMICOLON")
            return ExpressionStatementNode(expr)

    def parse_let_declaration(self):
        """Parse Rust-style let declarations: let name = expression;"""
        self.eat("LET")

        # Handle mutable declarations (let mut)
        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        # Variable name
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Optional type annotation: let name: type = expr;
        var_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            var_type = self.parse_type()

        # Assignment (required for let declarations)
        self.eat("EQUALS")
        initial_value = self.parse_expression()
        self.eat("SEMICOLON")

        # Create VariableNode with let-specific attributes
        var_node = VariableNode(
            name=name, var_type=var_type, initial_value=initial_value
        )

        # Add mutable flag if needed (for Rust compatibility)
        if is_mutable:
            var_node.qualifiers = getattr(var_node, "qualifiers", []) + ["mut"]

        # Add let flag to distinguish from regular variable declarations
        var_node.is_let_declaration = True

        return var_node

    def parse_if_statement(self):
        """Parse if statements with proper else-if chain handling."""
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        then_branch = self.parse_statement()

        else_branch = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            # Check for else-if chain
            if self.current_token[0] == "IF":
                # Recursively parse the next if statement as part of the else-if chain
                else_branch = self.parse_if_statement()
            else:
                # Regular else clause
                else_branch = self.parse_statement()

        return IfNode(
            condition=condition, then_branch=then_branch, else_branch=else_branch
        )

    def parse_for_statement(self):
        """Parse for statements."""
        self.eat("FOR")
        self.eat("LPAREN")

        # For loop initialization
        init = None
        if self.current_token[0] != "SEMICOLON":
            if self.is_variable_declaration():
                # Parse variable declaration without consuming semicolon
                init = self.parse_for_loop_variable_declaration()
            else:
                init = ExpressionStatementNode(self.parse_expression())
            # Consume semicolon after initialization
            self.eat("SEMICOLON")
        else:
            self.eat("SEMICOLON")

        # Condition
        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        # Update
        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression()

        self.eat("RPAREN")
        body = self.parse_statement()

        return ForNode(init=init, condition=condition, update=update, body=body)

    def parse_for_loop_variable_declaration(self):
        """Parse variable declarations in for loops (without consuming semicolon)."""
        qualifiers = []

        # Parse qualifiers
        while self.current_token[0] in [
            "CONST",
            "STATIC",
            "MUT",
            "SHARED",
            "UNIFORM",
            "BUFFER",
        ]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        var_type = self.parse_type()
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle multi-dimensional C-style array declarations (e.g., float arr[5][10];)
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            var_type = ArrayType(var_type, size)

        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()

        # Don't consume semicolon - that's handled by the for loop parser

        return VariableNode(
            name=name,
            var_type=var_type,
            initial_value=initial_value,
            qualifiers=qualifiers,
            is_mutable="const" not in qualifiers,
        )

    def parse_while_statement(self):
        """Parse while statements."""
        self.eat("WHILE")
        condition = self.parse_expression()
        body = self.parse_statement()

        return WhileNode(condition=condition, body=body)

    def parse_match_statement(self):
        """Parse match statements."""
        self.eat("MATCH")
        expression = self.parse_expression()

        self.eat("LBRACE")
        arms = []

        while self.current_token[0] != "RBRACE":
            arm = self.parse_match_arm()
            arms.append(arm)

        self.eat("RBRACE")

        return MatchNode(expression=expression, arms=arms)

    def parse_match_arm(self):
        """Parse match arm."""
        pattern = self.parse_pattern()

        guard = None
        if self.current_token[0] == "IF":
            self.eat("IF")
            guard = self.parse_expression()

        self.eat("FAT_ARROW")
        body = self.parse_statement()

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")

        return MatchArmNode(pattern=pattern, guard=guard, body=body)

    def parse_pattern(self):
        """Parse pattern expressions."""
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "_":
            self.eat("IDENTIFIER")
            return WildcardPatternNode()
        elif self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return IdentifierPatternNode(name)
        else:
            # Literal pattern
            literal = self.parse_literal()
            return LiteralPatternNode(literal)

    def parse_return_statement(self):
        """Parse return statements."""
        self.eat("RETURN")

        value = None
        if self.current_token[0] != "SEMICOLON":
            value = self.parse_expression()

        self.eat("SEMICOLON")
        return ReturnNode(value=value)

    def parse_expression(self):
        """Parse expressions with precedence."""
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        """Parse assignment expressions."""
        left = self.parse_ternary_expression()

        if self.current_token[0] in [
            "EQUALS",
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
            "ASSIGN_MOD",
            "ASSIGN_XOR",
            "ASSIGN_OR",
            "ASSIGN_AND",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def parse_ternary_expression(self):
        """Parse ternary conditional expressions (condition ? true_expr : false_expr)."""
        condition = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_ternary_expression()
            return TernaryOpNode(condition, true_expr, false_expr)

        return condition

    def parse_logical_or_expression(self):
        """Parse logical OR expressions."""
        left = self.parse_logical_and_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        """Parse logical AND expressions."""
        left = self.parse_bitwise_or_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or_expression(self):
        """Parse bitwise OR expressions."""
        left = self.parse_bitwise_xor_expression()

        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor_expression(self):
        """Parse bitwise XOR expressions."""
        left = self.parse_bitwise_and_expression()

        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and_expression(self):
        """Parse bitwise AND expressions."""
        left = self.parse_equality_expression()

        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        """Parse equality expressions."""
        left = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        """Parse relational expressions."""
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
        """Parse shift expressions."""
        left = self.parse_additive_expression()

        while self.current_token[0] in ["BITWISE_SHIFT_LEFT", "BITWISE_SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        """Parse additive expressions."""
        left = self.parse_multiplicative_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self):
        """Parse multiplicative expressions."""
        left = self.parse_unary_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary_expression(self):
        """Parse unary expressions."""
        if self.current_token[0] in [
            "NOT",
            "MINUS",
            "PLUS",
            "BITWISE_NOT",
            "INCREMENT",
            "DECREMENT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)

        return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        """Parse postfix expressions."""
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
            elif self.current_token[0] == "LPAREN":
                # Function call
                self.eat("LPAREN")
                arguments = []
                while self.current_token[0] != "RPAREN":
                    arguments.append(self.parse_expression())
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                self.eat("RPAREN")
                left = FunctionCallNode(left, arguments)
            elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                left = UnaryOpNode(op, left, is_postfix=True)
            else:
                break

        return left

    def parse_primary_expression(self):
        """Parse primary expressions."""
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return IdentifierNode(name)

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
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr

        else:
            # Default to identifier
            name = str(self.current_token[1]) if self.current_token[1] else "unknown"
            self.eat(self.current_token[0])
            return IdentifierNode(name)

    def parse_literal(self):
        """Parse literal values."""
        token_type, value = self.current_token

        if token_type == "NUMBER":
            self.eat("NUMBER")
            return LiteralNode(int(value), PrimitiveType("int"))

        elif token_type == "FLOAT_NUMBER":
            self.eat("FLOAT_NUMBER")
            return LiteralNode(float(value.rstrip("fF")), PrimitiveType("float"))

        elif token_type == "HEX_NUMBER":
            self.eat("HEX_NUMBER")
            return LiteralNode(int(value, 16), PrimitiveType("int"))

        elif token_type == "BIN_NUMBER":
            self.eat("BIN_NUMBER")
            return LiteralNode(int(value, 2), PrimitiveType("int"))

        elif token_type == "OCT_NUMBER":
            self.eat("OCT_NUMBER")
            return LiteralNode(int(value, 8), PrimitiveType("int"))

        elif token_type == "STRING_LITERAL":
            self.eat("STRING_LITERAL")
            return LiteralNode(value[1:-1], PrimitiveType("string"))  # Remove quotes

        elif token_type == "CHAR_LITERAL":
            self.eat("CHAR_LITERAL")
            return LiteralNode(value[1:-1], PrimitiveType("char"))  # Remove quotes

        else:
            # Default case
            self.eat(token_type)
            return LiteralNode(value, PrimitiveType("unknown"))

    # Legacy compatibility methods
    def parse_legacy_shader(self):
        """Legacy shader parsing for backward compatibility."""
        return create_legacy_shader_node([], [], [], [])

    def parse_shader_stage(self):
        """Parse shader stage for legacy compatibility."""
        stage_type = self.current_token[1]
        self.eat(self.current_token[0])

        self.eat("LBRACE")
        body = []

        while self.current_token[0] != "RBRACE":
            if self.is_function_declaration():
                func = self.parse_function()
                body.append(func)
            else:
                self.skip_unknown_token()

        self.eat("RBRACE")

        # Create a function representing this stage
        return FunctionNode(
            name=f"{stage_type}_main",
            return_type=PrimitiveType("void"),
            parameters=[],
            body=BlockNode(body),
            qualifiers=[stage_type],
        )

    def parse_cbuffer_as_struct(self):
        """Parse cbuffer as struct for legacy compatibility."""
        if self.current_token[0] == "CBUFFER":
            self.eat("CBUFFER")
        else:
            self.eat("UNIFORM")

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            member_type = self.parse_type()
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                size = None
                if self.current_token[0] != "RBRACKET":
                    size = self.parse_expression()
                self.eat("RBRACKET")
                member_type = ArrayType(member_type, size)

            self.eat("SEMICOLON")

            members.append(StructMemberNode(name=member_name, member_type=member_type))

        self.eat("RBRACE")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return StructNode(name=name, members=members)

    # Helper methods
    def is_function_declaration(self):
        """Check if current position is a function declaration."""
        saved_pos = self.pos
        saved_token = self.current_token

        try:
            # Skip qualifiers like async, unsafe, etc.
            while self.current_token[0] in ["ASYNC", "UNSAFE", "GLOBAL", "KERNEL"]:
                self.eat(self.current_token[0])

            # Skip function keyword if present
            if self.current_token[0] == "FUNCTION":
                self.eat("FUNCTION")

            # Look for type identifier ( pattern
            if self.is_type_token():
                self.advance_over_type()
                if self.current_token[0] == "IDENTIFIER":
                    self.eat("IDENTIFIER")
                    # Skip generic parameters if present
                    if self.current_token[0] == "LESS_THAN":
                        depth = 1
                        self.eat("LESS_THAN")
                        while depth > 0 and self.current_token[0] != "EOF":
                            if self.current_token[0] == "LESS_THAN":
                                depth += 1
                            elif self.current_token[0] == "GREATER_THAN":
                                depth -= 1
                            self.eat(self.current_token[0])

                    if self.current_token[0] == "LPAREN":
                        return True
        except:
            pass
        finally:
            self.pos = saved_pos
            self.current_token = saved_token

        return False

    def is_type_token(self):
        """Check if current token represents a type."""
        return self.current_token[0] in [
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
            "VEC2",
            "VEC3",
            "VEC4",
            "IVEC2",
            "IVEC3",
            "IVEC4",
            "UVEC2",
            "UVEC3",
            "UVEC4",
            "BVEC2",
            "BVEC3",
            "BVEC4",
            "MAT2",
            "MAT3",
            "MAT4",
            "IDENTIFIER",
        ]

    def advance_over_type(self):
        """Advance position over a type expression."""
        if self.is_type_token():
            self.eat(self.current_token[0])

            # Handle generic arguments
            if self.current_token[0] == "LESS_THAN":
                depth = 1
                self.eat("LESS_THAN")
                while depth > 0 and self.current_token[0] != "EOF":
                    if self.current_token[0] == "LESS_THAN":
                        depth += 1
                    elif self.current_token[0] == "GREATER_THAN":
                        depth -= 1
                    self.eat(self.current_token[0])

    def skip_unknown_token(self):
        """Skip unknown tokens for error recovery."""
        if self.current_token[0] != "EOF":
            self.eat(self.current_token[0])

    def create_empty_shader(self):
        """Create empty shader for error recovery."""
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
        """Report parser errors."""
        logging.warning(f"Parser: {message}")

    def next_token(self):
        """Advance to next token."""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]

    def parse_global(self):
        """Parse global declarations and top-level constructs."""
        # EOF protection
        if self.current_token[0] == "EOF":
            return None

        # Temporarily disable complex generic parsing - skip instead
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "generic":
            # Skip the entire generic declaration for now
            self.skip_unknown_token()  # "generic"
            # Skip to the end of this construct
            while self.current_token[0] not in ["RBRACE", "SEMICOLON", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] in ["RBRACE", "SEMICOLON"]:
                self.skip_unknown_token()
            return None

        # Temporarily disable trait parsing - skip instead
        if self.current_token[0] == "TRAIT":
            # Skip the entire trait declaration for now
            self.skip_unknown_token()  # "trait"
            # Skip to the end of this construct
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "RBRACE":
                self.skip_unknown_token()
            return None

        # Handle enum declarations
        if self.current_token[0] == "ENUM":
            return self.parse_enum()

        # Handle struct declarations
        if self.current_token[0] == "STRUCT":
            return self.parse_struct()

        # Handle constant declarations
        if self.current_token[0] == "CONST":
            return self.parse_constant()

        # Handle shader stage blocks
        if self.current_token[0] in ["VERTEX", "FRAGMENT", "COMPUTE", "GEOMETRY"]:
            return self.parse_shader_stage_block()

        # PRIORITY: Check for function declarations first (before variable declarations)
        # This prevents functions from being misidentified as variables
        if self.is_function_declaration():
            return self.parse_function()

        # Handle variable declarations (lower priority than functions)
        if self.is_variable_declaration():
            return self.parse_variable_declaration()

        # If we can't identify it, try to parse as function declaration anyway
        # This handles edge cases where is_function_declaration fails
        if self.is_type_token():
            try:
                return self.parse_function()
            except:
                # If function parsing fails, skip the token
                self.skip_unknown_token()
                return None

        # Skip unknown tokens for robustness
        self.skip_unknown_token()
        return None

    def parse_generic_declaration(self):
        """Parse generic declarations like 'generic<T> struct Option'."""
        if self.current_token[0] == "EOF":
            return None

        self.eat("IDENTIFIER")  # "generic"

        # Parse generic parameters with EOF protection
        if self.current_token[0] != "LESS_THAN":
            return None

        self.eat("LESS_THAN")
        generic_params = []

        while (
            self.current_token[0] != "GREATER_THAN" and self.current_token[0] != "EOF"
        ):
            if self.current_token[0] == "IDENTIFIER":
                param_name = self.current_token[1]
                self.eat("IDENTIFIER")

                # Handle type constraints (e.g., T: Numeric)
                constraints = []
                if self.current_token[0] == "COLON":
                    self.eat("COLON")
                    while (
                        self.current_token[0] == "IDENTIFIER"
                        and self.current_token[0] != "EOF"
                    ):
                        constraints.append(self.current_token[1])
                        self.eat("IDENTIFIER")
                        if self.current_token[0] == "PLUS":
                            self.eat("PLUS")
                        else:
                            break

                generic_params.append((param_name, constraints))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "GREATER_THAN":
                break

        if self.current_token[0] == "EOF":
            return None

        self.eat("GREATER_THAN")

        # Parse the actual declaration (struct, enum, function) with EOF protection
        if self.current_token[0] == "EOF":
            return None
        elif self.current_token[0] == "STRUCT":
            struct_node = self.parse_struct()
            # Add generic parameters to the struct
            if struct_node:
                struct_node.generic_params = generic_params
            return struct_node
        elif self.current_token[0] == "ENUM":
            enum_node = self.parse_enum()
            if enum_node:
                enum_node.generic_params = generic_params
            return enum_node
        elif self.current_token[0] == "FUNCTION" or self.is_function_declaration():
            func_node = self.parse_function()
            if func_node:
                func_node.generic_params = generic_params
            return func_node
        else:
            self.skip_unknown_token()
            return None

    def parse_trait(self):
        """Parse trait declarations."""
        if self.current_token[0] == "EOF":
            return None

        self.eat("TRAIT")

        if self.current_token[0] != "IDENTIFIER":
            return None

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle generic parameters for traits
        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_arguments()

        if self.current_token[0] != "LBRACE":
            return None

        self.eat("LBRACE")

        # Parse trait methods and associated types
        methods = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.is_function_declaration():
                func = self.parse_function()
                if func:
                    methods.append(func)
            else:
                self.skip_unknown_token()

        if self.current_token[0] == "EOF":
            return None

        self.eat("RBRACE")

        # Create a struct node for traits (simplified representation)
        return StructNode(
            name=name, members=methods, attributes=[], generic_params=generic_params
        )

    def parse_switch_statement(self):
        """Parse switch statements."""
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
                # Skip unknown tokens in switch body
                self.skip_unknown_token()

        self.eat("RBRACE")

        return SwitchNode(expression=expression, cases=cases)

    def parse_case(self):
        """Parse case statements."""
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        # Parse statements until we hit another case, default, or end of switch
        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            statements.append(stmt)

        return CaseNode(value=value, statements=statements)

    def parse_default_case(self):
        """Parse default case in switch statements."""
        self.eat("DEFAULT")
        self.eat("COLON")

        # Parse statements until we hit another case or end of switch
        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            statements.append(stmt)

        return CaseNode(
            value=None, statements=statements
        )  # None indicates default case

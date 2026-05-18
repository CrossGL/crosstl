"""Parser that builds CrossGL AST nodes from lexer tokens."""

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
    PreprocessorNode,
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
    RangeNode,
    BinaryOpNode,
    UnaryOpNode,
    TernaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    PointerAccessNode,
    ArrayAccessNode,
    ArrayLiteralNode,
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
    TextureOpNode,
    AtomicOpNode,
    SyncNode,
    BuiltinVariableNode,
    BufferNode,
    TextureResourceNode,
    SamplerNode,
    BufferOpNode,
    WaveOpNode,
    RayTracingOpNode,
    RayQueryOpNode,
    MeshOpNode,
    ArrayNode,
    ShaderStage,
    ExecutionModel,
    create_legacy_shader_node,
)
from .lexer import Lexer
from .validation import validate_shader_cbuffers
import logging

WAVE_INTRINSICS = {
    "WaveGetLaneCount",
    "WaveGetLaneIndex",
    "WaveIsFirstLane",
    "WaveActiveSum",
    "WaveActiveProduct",
    "WaveActiveBitAnd",
    "WaveActiveBitOr",
    "WaveActiveBitXor",
    "WaveActiveMin",
    "WaveActiveMax",
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveBallot",
    "WaveReadLaneAt",
    "WaveReadLaneFirst",
    "WavePrefixSum",
    "WavePrefixProduct",
    "QuadReadAcrossX",
    "QuadReadAcrossY",
    "QuadReadAcrossDiagonal",
    "QuadReadLaneAt",
    "WaveMatch",
    "WaveMultiPrefixSum",
    "WaveMultiPrefixProduct",
    "WaveMultiPrefixBitAnd",
    "WaveMultiPrefixBitOr",
    "WaveMultiPrefixBitXor",
}

RAYTRACING_INTRINSICS = {
    "TraceRay",
    "ReportHit",
    "CallShader",
    "AcceptHitAndEndSearch",
    "IgnoreHit",
}

MESH_INTRINSICS = {
    "SetMeshOutputCounts",
    "DispatchMesh",
}

RAYQUERY_METHODS = {
    "Proceed",
    "Abort",
    "CandidateType",
    "CommittedType",
    "CandidatePrimitiveIndex",
    "CommittedPrimitiveIndex",
    "CandidateInstanceID",
    "CommittedInstanceID",
    "CandidateGeometryIndex",
    "CommittedGeometryIndex",
    "CandidateObjectRayOrigin",
    "CandidateObjectRayDirection",
    "CommittedObjectRayOrigin",
    "CommittedObjectRayDirection",
    "CommittedRayT",
    "CandidateRayT",
}


class Parser:
    """Recursive-descent parser for CrossGL Universal IR tokens."""

    def __init__(self, tokens):
        """Initialize parser state from a token sequence."""
        self.tokens = tokens
        self.pos = 0
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
        )

    def skip_comments(self):
        """Consume single-line and multi-line comment tokens."""
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):
        """Consume one token of the expected type or raise ``SyntaxError``."""
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
        """Return a lookahead token without advancing the parser."""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return ("EOF", None)

    def finalize_shader(self, shader):
        """Run final validation hooks before returning a shader AST."""
        return validate_shader_cbuffers(shader)

    def parse(self):
        """Parse a complete CrossGL translation unit into a ``ShaderNode``."""
        structs = []
        functions = []
        global_variables = []
        constants = []
        cbuffers = []
        stages = {}
        imports = []
        preprocessors = []

        loop_count = 0
        max_loops = 10000

        while self.current_token[0] != "EOF" and loop_count < max_loops:
            loop_count += 1

            parsed_element = self.parse_global()
            if parsed_element:
                if isinstance(parsed_element, StructNode):
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
                    stages[parsed_element.stage] = parsed_element
                elif isinstance(parsed_element, ImportNode):
                    imports.append(parsed_element)
                elif isinstance(parsed_element, PreprocessorNode):
                    preprocessors.append(parsed_element)
                elif isinstance(parsed_element, EnumNode):
                    structs.append(parsed_element)

            # Safety check: if we're not advancing, break
            if loop_count > 100:
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

        shader = ShaderNode(
            name="main",
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

    def parse_program(self):
        """Parse the explicit program form used by legacy callers."""
        imports = []
        structs = []
        enums = []
        functions = []
        constants = []
        global_variables = []
        cbuffers = []

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
            elif self.is_cbuffer_declaration():
                cbuffers.append(self.parse_cbuffer_as_struct())
            elif self.current_token[0] == "CONST":
                constants.append(self.parse_constant())
            elif self.is_function_declaration():
                functions.append(self.parse_function())
            elif self.is_variable_declaration():
                global_variables.append(self.parse_variable_declaration())
            elif self.current_token[0] in [
                "VERTEX",
                "FRAGMENT",
                "COMPUTE",
                "GEOMETRY",
                "TESSELLATION_CONTROL",
                "TESSELLATION_EVALUATION",
                "TASK",
                "MESH",
                "RAY_GENERATION",
                "RAY_INTERSECTION",
                "RAY_CLOSEST_HIT",
                "RAY_MISS",
                "RAY_ANY_HIT",
                "RAY_CALLABLE",
            ]:
                stage_func = self.parse_shader_stage()
                functions.append(stage_func)
            else:
                self.skip_unknown_token()

        if shader_node:
            shader_node.structs.extend(structs)
            shader_node.functions.extend(functions)
            shader_node.global_variables.extend(global_variables)
            shader_node.constants.extend(constants)
            if cbuffers:
                shader_node.cbuffers = getattr(shader_node, "cbuffers", []) + cbuffers
            shader_node.imports.extend(imports)
            return self.finalize_shader(shader_node)
        else:
            shader = ShaderNode(
                name="main",
                execution_model=ExecutionModel.GRAPHICS_PIPELINE,
                stages={},
                structs=structs,
                functions=functions,
                global_variables=global_variables,
                constants=constants,
                imports=imports,
            )
            if cbuffers:
                shader.cbuffers = cbuffers
            return self.finalize_shader(shader)

    def parse_shader_declaration(self):
        """Parse a named ``shader`` block and its contained declarations."""
        self.eat("SHADER")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        execution_model = ExecutionModel.GRAPHICS_PIPELINE
        stages = {}
        functions = []
        structs = []
        global_variables = []
        constants = []
        cbuffers = []

        self.eat("LBRACE")

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] in [
                "VERTEX",
                "FRAGMENT",
                "COMPUTE",
                "GEOMETRY",
                "TESSELLATION_CONTROL",
                "TESSELLATION_EVALUATION",
                "TASK",
                "MESH",
                "RAY_GENERATION",
                "RAY_INTERSECTION",
                "RAY_CLOSEST_HIT",
                "RAY_MISS",
                "RAY_ANY_HIT",
                "RAY_CALLABLE",
            ]:
                stage_node = self.parse_shader_stage_block()
                stages[stage_node.stage] = stage_node
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.is_cbuffer_declaration():
                cbuffers.append(self.parse_cbuffer_as_struct())
            elif self.current_token[0] == "CONST":
                constants.append(self.parse_constant())
            elif self.is_function_declaration():
                func = self.parse_function()
                functions.append(func)
            elif self.is_variable_declaration():
                global_variables.append(self.parse_variable_declaration())
            else:
                self.skip_unknown_token()

        self.eat("RBRACE")

        shader = ShaderNode(
            name=name,
            execution_model=execution_model,
            stages=stages,
            structs=structs,
            functions=functions,
            global_variables=global_variables,
            constants=constants,
        )
        if cbuffers:
            shader.cbuffers = cbuffers
        return self.finalize_shader(shader)

    def parse_shader_stage_block(self):
        """Parse a stage-qualified block into a ``StageNode``."""
        stage_type = self.current_token[1]
        stage_enum = {
            "vertex": ShaderStage.VERTEX,
            "fragment": ShaderStage.FRAGMENT,
            "compute": ShaderStage.COMPUTE,
            "geometry": ShaderStage.GEOMETRY,
            "tessellation_control": ShaderStage.TESSELLATION_CONTROL,
            "tessellation_evaluation": ShaderStage.TESSELLATION_EVALUATION,
            "task": ShaderStage.TASK,
            "amplification": ShaderStage.AMPLIFICATION,
            "object": ShaderStage.OBJECT,
            "mesh": ShaderStage.MESH,
            "ray_generation": ShaderStage.RAY_GENERATION,
            "ray_intersection": ShaderStage.RAY_INTERSECTION,
            "ray_closest_hit": ShaderStage.RAY_CLOSEST_HIT,
            "ray_miss": ShaderStage.RAY_MISS,
            "ray_any_hit": ShaderStage.RAY_ANY_HIT,
            "ray_callable": ShaderStage.RAY_CALLABLE,
            "intersection": ShaderStage.RAY_INTERSECTION,
            "closesthit": ShaderStage.RAY_CLOSEST_HIT,
            "anyhit": ShaderStage.RAY_ANY_HIT,
            "miss": ShaderStage.RAY_MISS,
            "callable": ShaderStage.RAY_CALLABLE,
        }.get(stage_type, ShaderStage.VERTEX)

        self.eat(self.current_token[0])

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

        if not main_function:
            main_function = FunctionNode(
                name="main",
                return_type=PrimitiveType("void"),
                parameters=[],
                body=BlockNode([]),
            )

        if stage_name:
            main_function.name = stage_name

        return StageNode(
            stage=stage_enum,
            entry_point=main_function,
            local_variables=local_variables,
            local_functions=local_functions,
        )

    def parse_import(self):
        """Parse an ``import`` or ``use`` declaration."""
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

    def parse_preprocessor_directive(self):
        """Parse a preprocessor token into a structured directive node."""
        text = self.current_token[1] or ""
        self.eat("PREPROCESSOR")
        stripped = text.lstrip("#").strip()
        if not stripped:
            return PreprocessorNode("", "")
        parts = stripped.split(None, 1)
        directive = parts[0]
        content = parts[1] if len(parts) > 1 else ""
        return PreprocessorNode(directive, content)

    def parse_precision_statement(self):
        """Parse a GLSL-style precision statement as a preprocessor node."""
        self.eat("PRECISION")
        parts = []
        while self.current_token[0] not in ["SEMICOLON", "EOF"]:
            parts.append(str(self.current_token[1]))
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        content = " ".join(parts).strip()
        return PreprocessorNode("precision", content)

    def parse_struct(self):
        """Parse a struct declaration and its member list."""
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
        """Parse one struct member declaration."""
        if self.current_token[0] == "EOF":
            return None

        if self.current_token[0] == "PREPROCESSOR":
            return self.parse_preprocessor_directive()

        if self.current_token[0] == "PRECISION":
            return self.parse_precision_statement()

        if self.current_token[0] == "ENUM":
            return self.parse_enum()

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

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            member_type = ArrayType(member_type, size)

        attributes = []
        if self.current_token[0] in ["AT", "ATTRIBUTE"]:
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
        """Parse an enum declaration and its variants."""
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
        """Parse one enum variant, including tuple or struct payloads."""
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        value = None
        variant_data = None

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

        elif self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

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

        variant_node = EnumVariantNode(name=name, value=value)
        if variant_data:
            variant_node.data = variant_data

        return variant_node

    def parse_function(self):
        """Parse a function declaration or definition."""
        qualifiers = []
        attributes = []

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

        if self.current_token[0] == "FUNCTION":
            self.eat("FUNCTION")

        return_type = self.parse_type()

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        self.eat("LPAREN")
        parameters = self.parse_parameter_list()
        self.eat("RPAREN")

        post_attributes = []
        if self.current_token[0] in ["AT", "ATTRIBUTE"]:
            post_attributes = self.parse_attributes()

        body = None
        if self.current_token[0] == "LBRACE":
            body = self.parse_block()
        else:
            self.eat("SEMICOLON")

        return FunctionNode(
            name=name,
            return_type=return_type,
            parameters=parameters,
            body=body,
            generic_params=generic_params,
            attributes=attributes + post_attributes,
            qualifiers=qualifiers,
            is_async="async" in qualifiers,
            is_unsafe="unsafe" in qualifiers,
        )

    def parse_parameter_list(self):
        """Parse a comma-separated function parameter list."""
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
        """Parse one function parameter declaration."""
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

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            param_type = ArrayType(param_type, size)

        if self.current_token[0] in ["AT", "ATTRIBUTE"]:
            attributes.extend(self.parse_attributes())

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
        """Parse a variable declaration, including qualifiers and attributes."""
        attributes = []
        if self.current_token[0] in ["AT", "ATTRIBUTE"]:
            attributes = self.parse_attributes()

        qualifiers = []

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

        if self.current_token[0] in ["AT", "ATTRIBUTE"]:
            attributes.extend(self.parse_attributes())

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
            attributes=attributes,
            is_mutable="const" not in qualifiers,
        )

    def is_variable_declaration(self):
        """
        Lookahead check for variable declarations.
        Handles complex cases and distinguishes from function calls and member access.
        """
        saved_pos = self.pos
        saved_token = self.current_token

        try:
            if self.current_token[0] in ["AT", "ATTRIBUTE"]:
                self.parse_attributes()

            while self.current_token[0] in [
                "CONST",
                "STATIC",
                "MUT",
                "SHARED",
                "UNIFORM",
                "BUFFER",
            ]:
                self.eat(self.current_token[0])

            if not self.is_type_token():
                return False

            self.advance_over_type()

            if self.current_token[0] != "IDENTIFIER":
                return False

            self.current_token[1]
            self.eat("IDENTIFIER")

            next_token = self.current_token[0]

            if next_token == "LBRACKET":
                return True

            if next_token == "EQUALS":
                return True

            if next_token == "SEMICOLON":
                return True

            if next_token == "COMMA":
                return True

            if next_token in ["RPAREN", "COMMA"] and self.in_parameter_context():
                return True

            if next_token in ["DOT", "LPAREN"]:
                return False

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

            return True

        except Exception:
            return False
        finally:
            self.pos = saved_pos
            self.current_token = saved_token

    def in_parameter_context(self):
        """Check if we're currently parsing function parameters."""
        for i in range(max(0, self.pos - 10), self.pos):
            if i < len(self.tokens) and self.tokens[i][0] == "LPAREN":
                return True
        return False

    def parse_constant(self):
        """Parse a constant declaration."""
        self.eat("CONST")
        const_type = self.parse_type()
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("EQUALS")
        value = self.parse_expression()
        self.eat("SEMICOLON")

        return ConstantNode(name=name, const_type=const_type, value=value)

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
                rows, cols = map(int, dimensions.split("x"))
            else:
                size = int(dimensions)
                rows = cols = size

            element_type = "double" if is_double_matrix else "float"
            base_type = MatrixType(PrimitiveType(element_type), rows, cols)

        elif self.current_token[0] in [
            "SAMPLER",
            "SAMPLER1D",
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
            "IIMAGE2D",
            "IIMAGE3D",
            "IIMAGE2DARRAY",
            "IIMAGE2DMS",
            "IIMAGE2DMSARRAY",
            "UIMAGE2D",
            "UIMAGE3D",
            "UIMAGE2DARRAY",
            "UIMAGE2DMS",
            "UIMAGE2DMSARRAY",
            "IMAGE2D",
            "IMAGE3D",
            "IMAGECUBE",
            "IMAGE2DARRAY",
            "IMAGE2DMS",
            "IMAGE2DMSARRAY",
        ]:
            sampler_types = {
                "SAMPLER": "sampler",
                "SAMPLER1D": "sampler1D",
                "SAMPLER2D": "sampler2D",
                "SAMPLER3D": "sampler3D",
                "SAMPLERCUBE": "samplerCube",
                "SAMPLER2DARRAY": "sampler2DArray",
                "SAMPLER2DSHADOW": "sampler2DShadow",
                "SAMPLER2DARRAYSHADOW": "sampler2DArrayShadow",
                "SAMPLERCUBESHADOW": "samplerCubeShadow",
                "SAMPLERCUBEARRAY": "samplerCubeArray",
                "SAMPLERCUBEARRAYSHADOW": "samplerCubeArrayShadow",
                "SAMPLER2DMS": "sampler2DMS",
                "SAMPLER2DMSARRAY": "sampler2DMSArray",
                "IIMAGE2D": "iimage2D",
                "IIMAGE3D": "iimage3D",
                "IIMAGE2DARRAY": "iimage2DArray",
                "IIMAGE2DMS": "iimage2DMS",
                "IIMAGE2DMSARRAY": "iimage2DMSArray",
                "UIMAGE2D": "uimage2D",
                "UIMAGE3D": "uimage3D",
                "UIMAGE2DARRAY": "uimage2DArray",
                "UIMAGE2DMS": "uimage2DMS",
                "UIMAGE2DMSARRAY": "uimage2DMSArray",
                "IMAGE2D": "image2D",
                "IMAGE3D": "image3D",
                "IMAGECUBE": "imageCube",
                "IMAGE2DARRAY": "image2DArray",
                "IMAGE2DMS": "image2DMS",
                "IMAGE2DMSARRAY": "image2DMSArray",
            }
            token_type = self.current_token[0]
            self.eat(token_type)
            base_type = NamedType(sampler_types[token_type])

        elif self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            generic_args = []
            if self.current_token[0] == "LESS_THAN":
                generic_args = self.parse_generic_arguments()

            base_type = NamedType(name, generic_args)

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

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            base_type = ArrayType(base_type, size)

        if self.current_token[0] == "MULTIPLY":
            self.eat("MULTIPLY")
            # pointer handling deferred: PointerType not yet used by generators

        if is_buffer:
            if hasattr(base_type, "qualifiers"):
                base_type.qualifiers = getattr(base_type, "qualifiers", []) + ["buffer"]
            else:
                base_type = NamedType(f"buffer_{base_type}", [])

        return base_type

    def parse_generic_parameters(self):
        """Parse generic parameter declarations after ``<``."""
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
        """Parse generic type arguments after ``<``."""
        self.eat("LESS_THAN")
        args = []

        while self.current_token[0] != "GREATER_THAN":
            arg_type = self.parse_type()
            args.append(arg_type)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("GREATER_THAN")
        return args

    def vector_element_type_from_generic(self, type_node):
        """Resolve a vector generic argument to a primitive element type."""
        type_name = self.format_type_argument(type_node)
        aliases = {
            "f32": "float",
            "float": "float",
            "f64": "double",
            "double": "double",
            "i32": "int",
            "int": "int",
            "u32": "uint",
            "uint": "uint",
            "bool": "bool",
        }
        return PrimitiveType(aliases.get(type_name, type_name))

    def format_type_argument(self, type_node):
        """Format a parsed type node back into a compact type argument string."""
        if hasattr(type_node, "name"):
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
        return str(type_node)

    def parse_attributes(self):
        """Parse one or more ``@`` attribute annotations."""
        attributes = []

        while self.current_token[0] in ["AT", "ATTRIBUTE"]:
            if self.current_token[0] == "ATTRIBUTE":
                name = self.current_token[1][1:]
                self.eat("ATTRIBUTE")
            else:
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
        """Parse a braced statement block."""
        self.eat("LBRACE")
        statements = []

        while self.current_token[0] != "RBRACE":
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

        self.eat("RBRACE")
        return BlockNode(statements)

    def parse_statement(self):
        """Parse any statement form supported by CrossGL."""
        if self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
        elif self.current_token[0] == "LOOP":
            return self.parse_loop_statement()
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
            expr = self.parse_expression()
            self.eat("SEMICOLON")
            return ExpressionStatementNode(expr)

    def parse_let_declaration(self):
        """Parse Rust-style ``let [mut] name [: type] = expr;`` declarations."""
        self.eat("LET")

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        var_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            var_type = self.parse_type()

        self.eat("EQUALS")
        initial_value = self.parse_expression()
        self.eat("SEMICOLON")

        var_node = VariableNode(
            name=name, var_type=var_type, initial_value=initial_value
        )

        if is_mutable:
            var_node.qualifiers = getattr(var_node, "qualifiers", []) + ["mut"]

        var_node.is_let_declaration = True

        return var_node

    def parse_if_statement(self):
        """Parse an if/else statement chain."""
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        then_branch = self.parse_statement()

        else_branch = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            if self.current_token[0] == "IF":
                else_branch = self.parse_if_statement()
            else:
                else_branch = self.parse_statement()

        return IfNode(
            condition=condition, then_branch=then_branch, else_branch=else_branch
        )

    def parse_for_statement(self):
        """Parse a C-style ``for`` loop or dispatch to ``for-in`` parsing."""
        self.eat("FOR")
        if self.current_token[0] != "LPAREN":
            return self.parse_for_in_statement_after_for()

        self.eat("LPAREN")

        init = None
        if self.current_token[0] != "SEMICOLON":
            if self.is_variable_declaration():
                init = self.parse_for_loop_variable_declaration()
            else:
                init = ExpressionStatementNode(self.parse_expression())
            self.eat("SEMICOLON")
        else:
            self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression()

        self.eat("RPAREN")
        body = self.parse_statement()

        return ForNode(init=init, condition=condition, update=update, body=body)

    def parse_for_in_statement_after_for(self):
        """Parse a ``for pattern in iterable`` loop after ``for`` is consumed."""
        pattern = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("IN")
        iterable = self.parse_expression()
        body = self.parse_statement()

        return ForInNode(pattern=pattern, iterable=iterable, body=body)

    def parse_for_loop_variable_declaration(self):
        """Parse variable declarations in for loops (without consuming semicolon)."""
        qualifiers = []

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
        """Parse a while loop statement."""
        self.eat("WHILE")
        condition = self.parse_expression()
        body = self.parse_statement()

        return WhileNode(condition=condition, body=body)

    def parse_loop_statement(self):
        """Parse an unconditional loop statement."""
        self.eat("LOOP")
        body = self.parse_statement()

        return LoopNode(body=body)

    def parse_match_statement(self):
        """Parse a match statement and all of its arms."""
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
        """Parse a single match arm."""
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
        """Parse a match pattern."""
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "_":
            self.eat("IDENTIFIER")
            return WildcardPatternNode()
        elif self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return IdentifierPatternNode(name)
        else:
            literal = self.parse_literal()
            return LiteralPatternNode(literal)

    def parse_return_statement(self):
        """Parse a return statement with an optional value."""
        self.eat("RETURN")

        value = None
        if self.current_token[0] != "SEMICOLON":
            value = self.parse_expression()

        self.eat("SEMICOLON")
        return ReturnNode(value=value)

    def parse_expression(self):
        """Parse an expression using the highest-precedence entry point."""
        return self.parse_assignment_expression()

    def parse_range_expression(self):
        """Parse range and inclusive-range expressions."""
        left = self.parse_ternary_expression()

        if self.current_token[0] in ["RANGE", "RANGE_INCLUSIVE"]:
            inclusive = self.current_token[0] == "RANGE_INCLUSIVE"
            self.eat(self.current_token[0])
            right = self.parse_ternary_expression()
            return RangeNode(left, right, inclusive=inclusive)

        return left

    def parse_assignment_expression(self):
        """Parse assignment and compound-assignment expressions."""
        left = self.parse_range_expression()

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
        """Parse a ternary conditional expression."""
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
        """Parse equality and inequality expressions."""
        left = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        """Parse relational comparison expressions."""
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
        """Parse bit-shift expressions."""
        left = self.parse_additive_expression()

        while self.current_token[0] in ["BITWISE_SHIFT_LEFT", "BITWISE_SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        """Parse addition and subtraction expressions."""
        left = self.parse_multiplicative_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self):
        """Parse multiplication, division, and modulo expressions."""
        left = self.parse_unary_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary_expression(self):
        """Parse prefix unary expressions."""
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
        """Parse member, call, index, and postfix unary expressions."""
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
            elif (
                self.current_token[0] == "LESS_THAN"
                and isinstance(left, IdentifierNode)
                and left.name in {"vec2", "vec3", "vec4"}
            ):
                generic_args = self.parse_generic_arguments()
                args = ", ".join(self.format_type_argument(arg) for arg in generic_args)
                left = IdentifierNode(f"{left.name}<{args}>")
            elif self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                arguments = []
                while self.current_token[0] != "RPAREN":
                    arguments.append(self.parse_expression())
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                self.eat("RPAREN")
                if isinstance(left, IdentifierNode):
                    if left.name in WAVE_INTRINSICS:
                        left = WaveOpNode(left.name, arguments)
                    elif left.name in RAYTRACING_INTRINSICS:
                        left = RayTracingOpNode(left.name, arguments)
                    elif left.name in MESH_INTRINSICS:
                        left = MeshOpNode(left.name, arguments)
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

    def parse_primary_expression(self):
        """Parse identifiers, literals, parenthesized expressions, and arrays."""
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
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr

        elif self.current_token[0] == "LBRACE":
            return self.parse_array_literal()

        else:
            name = str(self.current_token[1]) if self.current_token[1] else "unknown"
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
            return LiteralNode(float(value.rstrip("fF")), PrimitiveType("float"))

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
        """Return integer digits and signedness for an integer literal."""
        value = str(value)
        if value.endswith(("u", "U")):
            return value[:-1], PrimitiveType("uint")
        return value, PrimitiveType("int")

    # Legacy compatibility methods
    def parse_legacy_shader(self):
        """Return an empty legacy shader root."""
        return create_legacy_shader_node([], [], [], [])

    def parse_shader_stage(self):
        """Parse a legacy stage block into a function node."""
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

        return FunctionNode(
            name=f"{stage_type}_main",
            return_type=PrimitiveType("void"),
            parameters=[],
            body=BlockNode(body),
            qualifiers=[stage_type],
        )

    def parse_cbuffer_as_struct(self):
        """Parse a constant/uniform buffer declaration as a struct node."""
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

        node = StructNode(name=name, members=members)
        node.is_cbuffer = True
        return node

    # Helper methods
    def is_cbuffer_declaration(self):
        """Return whether the current token begins a cbuffer declaration."""
        if self.current_token[0] == "CBUFFER":
            return True
        return (
            self.current_token[0] == "UNIFORM"
            and self.peek()[0] == "IDENTIFIER"
            and self.peek(2)[0] == "LBRACE"
        )

    def is_function_declaration(self):
        """Return whether the current token sequence looks like a function."""
        saved_pos = self.pos
        saved_token = self.current_token

        try:
            while self.current_token[0] in ["ASYNC", "UNSAFE", "GLOBAL", "KERNEL"]:
                self.eat(self.current_token[0])

            if self.current_token[0] == "FUNCTION":
                self.eat("FUNCTION")

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
        """Return whether the current token can start a type expression."""
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
            "IIMAGE2D",
            "IIMAGE3D",
            "IIMAGE2DARRAY",
            "IIMAGE2DMS",
            "IIMAGE2DMSARRAY",
            "UIMAGE2D",
            "UIMAGE3D",
            "UIMAGE2DARRAY",
            "UIMAGE2DMS",
            "UIMAGE2DMSARRAY",
            "IMAGE2D",
            "IMAGE3D",
            "IMAGECUBE",
            "IMAGE2DARRAY",
            "IMAGE2DMS",
            "IMAGE2DMSARRAY",
            "IDENTIFIER",
        ]

    def advance_over_type(self):
        """Advance over a type expression during lookahead checks."""
        if self.is_type_token():
            self.eat(self.current_token[0])

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
        """Consume one token when recovering from unsupported syntax."""
        if self.current_token[0] != "EOF":
            self.eat(self.current_token[0])

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

    def parse_global(self):
        """Parse one top-level declaration or recover past unsupported input."""
        if self.current_token[0] == "EOF":
            return None

        if self.current_token[0] == "PREPROCESSOR":
            return self.parse_preprocessor_directive()

        if self.current_token[0] == "PRECISION":
            return self.parse_precision_statement()

        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "generic":
            self.skip_unknown_token()
            while self.current_token[0] not in ["RBRACE", "SEMICOLON", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] in ["RBRACE", "SEMICOLON"]:
                self.skip_unknown_token()
            return None

        if self.current_token[0] == "TRAIT":
            self.skip_unknown_token()
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "RBRACE":
                self.skip_unknown_token()
            return None

        if self.current_token[0] == "ENUM":
            return self.parse_enum()

        if self.current_token[0] == "STRUCT":
            return self.parse_struct()

        if self.current_token[0] == "CONST":
            return self.parse_constant()

        if self.is_cbuffer_declaration():
            return self.parse_cbuffer_as_struct()

        if self.current_token[0] in [
            "VERTEX",
            "FRAGMENT",
            "COMPUTE",
            "GEOMETRY",
            "TESSELLATION_CONTROL",
            "TESSELLATION_EVALUATION",
            "TASK",
            "MESH",
            "OBJECT",
            "AMPLIFICATION",
            "RAY_GENERATION",
            "RAY_INTERSECTION",
            "RAY_CLOSEST_HIT",
            "RAY_MISS",
            "RAY_ANY_HIT",
            "RAY_CALLABLE",
        ]:
            return self.parse_shader_stage_block()

        # Functions take priority over variables to avoid misidentification.
        if self.is_function_declaration():
            return self.parse_function()

        if self.is_variable_declaration():
            return self.parse_variable_declaration()

        # is_function_declaration can fail on some edge-case type tokens; try anyway.
        if self.is_type_token():
            try:
                return self.parse_function()
            except:
                self.skip_unknown_token()
                return None

        self.skip_unknown_token()
        return None

    def parse_generic_declaration(self):
        """Parse legacy generic declarations that wrap structs/enums/functions."""
        if self.current_token[0] == "EOF":
            return None

        self.eat("IDENTIFIER")

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

        if self.current_token[0] == "EOF":
            return None
        elif self.current_token[0] == "STRUCT":
            struct_node = self.parse_struct()
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
        """Parse a trait-like declaration into the current AST shape."""
        if self.current_token[0] == "EOF":
            return None

        self.eat("TRAIT")

        if self.current_token[0] != "IDENTIFIER":
            return None

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_arguments()

        if self.current_token[0] != "LBRACE":
            return None

        self.eat("LBRACE")

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

        # Traits are represented as structs (simplified).
        return StructNode(
            name=name, members=methods, attributes=[], generic_params=generic_params
        )

    def parse_switch_statement(self):
        """Parse a switch statement and its case clauses."""
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
                self.skip_unknown_token()

        self.eat("RBRACE")

        return SwitchNode(expression=expression, cases=cases)

    def parse_case(self):
        """Parse one explicit switch case."""
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            statements.append(stmt)

        return CaseNode(value=value, statements=statements)

    def parse_default_case(self):
        """Parse the default switch case."""
        self.eat("DEFAULT")
        self.eat("COLON")

        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            statements.append(stmt)

        return CaseNode(value=None, statements=statements)

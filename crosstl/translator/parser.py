"""Parser that builds CrossGL AST nodes from lexer tokens."""

import logging
from copy import deepcopy

from .ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CaseNode,
    ConstantNode,
    ConstructorNode,
    ConstructorPatternNode,
    ContinueNode,
    DoWhileNode,
    EnumNode,
    EnumVariantNode,
    ExecutionModel,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    FunctionType,
    GenericParameterNode,
    IdentifierNode,
    IdentifierPatternNode,
    IfNode,
    ImportNode,
    LayoutQualifierNode,
    LiteralNode,
    LiteralPatternNode,
    LoopNode,
    MatchArmNode,
    MatchNode,
    MatrixType,
    MemberAccessNode,
    MeshOpNode,
    NamedType,
    ParameterNode,
    PointerAccessNode,
    PointerType,
    PreprocessorNode,
    PrimitiveType,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StageMap,
    StageNode,
    StructMemberNode,
    StructNode,
    StructPatternNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
    WaveOpNode,
    WhileNode,
    WildcardPatternNode,
    create_legacy_shader_node,
)
from .stage_utils import shader_stage_from_name
from .validation import validate_shader_cbuffers

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
    "WaveActiveAllEqual",
    "WaveActiveBallot",
    "WaveActiveCountBits",
    "WaveReadLaneAt",
    "WaveReadLaneFirst",
    "WavePrefixSum",
    "WavePrefixProduct",
    "WavePrefixCountBits",
    "QuadReadAcrossX",
    "QuadReadAcrossY",
    "QuadReadAcrossDiagonal",
    "QuadReadLaneAt",
    "QuadAny",
    "QuadAll",
    "WaveMatch",
    "WaveMultiPrefixSum",
    "WaveMultiPrefixCountBits",
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

VARIABLE_QUALIFIER_TOKEN_TYPES = frozenset(
    {
        "CONST",
        "STATIC",
        "MUT",
        "SHARED",
        "UNIFORM",
        "BUFFER",
        "IN",
        "PRIVATE",
        "GLOBAL",
        "LOCAL",
        "THREADGROUP_IMAGEBLOCK",
        "THREADGROUP",
        "WORKGROUP",
    }
)

PARAMETER_QUALIFIER_TOKEN_TYPES = VARIABLE_QUALIFIER_TOKEN_TYPES - {"MUT"}

VARIABLE_QUALIFIER_NAMES = frozenset(
    {
        "out",
        "input",
        "output",
        "inout",
        "patch",
        "flat",
        "smooth",
        "noperspective",
        "centroid",
        "sample",
        "pervertex",
        "perprimitive",
        "perview",
        "readonly",
        "writeonly",
        "readwrite",
        "coherent",
        "volatile",
        "restrict",
        "device",
        "constant",
        "thread",
        "threadgroup_imageblock",
        "threadgroup",
        "workgroup",
        "storage",
        "private",
        "function",
        "groupshared",
        "row_major",
        "column_major",
        "precise",
        "nointerpolation",
        "linear",
        "linear_centroid",
        "linear_noperspective",
        "linear_noperspective_centroid",
        "linear_sample",
    }
)

PARAMETER_PRIMITIVE_QUALIFIER_NAMES = frozenset(
    {
        "point",
        "line",
        "triangle",
        "lineadj",
        "triangleadj",
    }
)

TEXTURE_TYPE_NAMES = {
    "TEXTURE1D": "texture1D",
    "TEXTURE2D": "texture2D",
    "TEXTURE3D": "texture3D",
    "TEXTURECUBE": "textureCube",
    "TEXTURE2DARRAY": "texture2DArray",
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
    "TraceRayInline",
}

SHADER_STAGE_TOKEN_TYPES = frozenset(
    {
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
    }
)


class Parser:
    """Recursive-descent parser for CrossGL Universal IR tokens."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
        )

    def skip_comments(self):
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

    def current_token_is_type_greater_than(self):
        """Return whether current token can close a generic type argument."""
        return self.current_token[0] in {"GREATER_THAN", "BITWISE_SHIFT_RIGHT"}

    def eat_type_greater_than(self):
        """Consume one generic ``>`` close, splitting lexer ``>>`` when needed."""
        if self.current_token[0] == "BITWISE_SHIFT_RIGHT":
            self.tokens[self.pos] = ("GREATER_THAN", ">")
            self.tokens.insert(self.pos + 1, ("GREATER_THAN", ">"))
            self.current_token = self.tokens[self.pos]
            self.eat("GREATER_THAN")
            return
        self.eat("GREATER_THAN")

    def peek(self, offset=1):
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return ("EOF", None)

    def finalize_shader(self, shader):
        return validate_shader_cbuffers(shader)

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

        loop_count = 0
        max_loops = 10000

        while self.current_token[0] != "EOF" and loop_count < max_loops:
            loop_count += 1
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
                current_token_info = (
                    f"{self.current_token[0]}:{self.current_token[1]}"
                    if self.current_token[1]
                    else self.current_token[0]
                )
                print(
                    f"Warning: Parser may be stuck at token {current_token_info}, breaking..."
                )
                self.skip_unknown_token()

        if loop_count >= max_loops:
            print(f"Warning: Parser hit maximum loop limit ({max_loops}), stopping...")

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

    def parse_program(self):
        """Parse the explicit program form used by legacy callers."""
        imports = []
        structs = []
        enums = []
        functions = []
        constants = []
        global_variables = []
        cbuffers = []
        preprocessors = []

        shader_node = None

        while self.current_token[0] != "EOF":
            if self.current_token[0] in ["IMPORT", "USE", "FROM"]:
                imports.append(self.parse_import())
            elif self.current_token[0] == "PREPROCESSOR":
                preprocessors.append(self.parse_preprocessor_directive())
            elif self.current_token[0] == "PRECISION":
                preprocessors.append(self.parse_precision_statement())
            elif self.current_token[0] == "SHADER":
                shader_node = self.parse_shader_declaration()
            elif self.current_token_starts_attributed_cbuffer_declaration():
                cbuffers.append(self.parse_attributed_cbuffer())
            elif self.current_token_starts_attributed_struct_declaration():
                structs.append(self.parse_attributed_struct())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "ENUM":
                enums.append(self.parse_enum())
            elif self.is_cbuffer_declaration():
                cbuffers.append(self.parse_cbuffer_as_struct())
            elif self.current_token[0] == "CONST":
                constants.append(self.parse_constant())
            elif self.current_token[0] == "LET":
                global_variables.append(self.parse_let_declaration())
            elif self.current_token[0] == "LAYOUT":
                attributes = self.parse_layout_attributes()
                if self.is_layout_buffer_block_declaration():
                    struct_node, variable_node = self.parse_layout_buffer_block(
                        attributes
                    )
                    structs.append(struct_node)
                    global_variables.append(variable_node)
                elif self.current_token_starts_attributed_cbuffer_declaration():
                    cbuffers.append(self.parse_attributed_cbuffer(attributes))
                elif self.is_cbuffer_declaration():
                    cbuffers.append(self.parse_cbuffer_as_struct(attributes))
                elif self.is_variable_declaration():
                    self.append_parsed_nodes(
                        global_variables, self.parse_variable_declaration(attributes)
                    )
                else:
                    self.skip_unknown_token()
            elif self.is_function_declaration():
                functions.append(self.parse_function())
            elif self.is_variable_declaration():
                self.append_parsed_nodes(
                    global_variables, self.parse_variable_declaration()
                )
            elif self.current_token[0] in SHADER_STAGE_TOKEN_TYPES:
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
            shader_node.preprocessors = preprocessors + getattr(
                shader_node, "preprocessors", []
            )
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
                preprocessors=preprocessors,
            )
            if cbuffers:
                shader.cbuffers = cbuffers
            return self.finalize_shader(shader)

    def parse_shader_declaration(self):
        """Parse a named ``shader`` block and its contained declarations."""
        self.eat("SHADER")
        name = self.parse_binding_identifier()

        execution_model = ExecutionModel.GRAPHICS_PIPELINE
        stages = StageMap()
        functions = []
        structs = []
        global_variables = []
        constants = []
        cbuffers = []
        preprocessors = []

        self.eat("LBRACE")

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "PREPROCESSOR":
                preprocessors.append(self.parse_preprocessor_directive())
            elif self.current_token[0] == "PRECISION":
                preprocessors.append(self.parse_precision_statement())
            elif self.current_token_starts_shader_stage_block():
                stage_node = self.parse_shader_stage_block()
                stages.append(stage_node.stage, stage_node)
            elif self.current_token_starts_attributed_cbuffer_declaration():
                cbuffers.append(self.parse_attributed_cbuffer())
            elif self.current_token_starts_attributed_struct_declaration():
                structs.append(self.parse_attributed_struct())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.is_generic_declaration():
                declaration = self.parse_generic_declaration()
                self.append_shader_declaration(
                    declaration,
                    structs,
                    functions,
                    global_variables,
                    constants,
                    cbuffers,
                    stages,
                )
            elif self.current_token[0] == "ENUM":
                structs.append(self.parse_enum())
            elif self.current_token[0] == "TRAIT":
                trait_node = self.parse_trait()
                self.append_shader_declaration(
                    trait_node,
                    structs,
                    functions,
                    global_variables,
                    constants,
                    cbuffers,
                    stages,
                )
            elif self.is_cbuffer_declaration():
                cbuffers.append(self.parse_cbuffer_as_struct())
            elif self.current_token[0] == "CONST":
                constants.append(self.parse_constant())
            elif self.current_token[0] == "LET":
                global_variables.append(self.parse_let_declaration())
            elif self.current_token[0] == "LAYOUT":
                attributes = self.parse_layout_attributes()
                if self.is_layout_buffer_block_declaration():
                    struct_node, variable_node = self.parse_layout_buffer_block(
                        attributes
                    )
                    structs.append(struct_node)
                    global_variables.append(variable_node)
                elif self.current_token_starts_attributed_cbuffer_declaration():
                    cbuffers.append(self.parse_attributed_cbuffer(attributes))
                elif self.is_cbuffer_declaration():
                    cbuffers.append(self.parse_cbuffer_as_struct(attributes))
                elif self.is_variable_declaration():
                    self.append_parsed_nodes(
                        global_variables, self.parse_variable_declaration(attributes)
                    )
                else:
                    self.skip_unknown_token()
            elif self.is_function_declaration():
                func = self.parse_function()
                functions.append(func)
            elif self.is_variable_declaration():
                self.append_parsed_nodes(
                    global_variables, self.parse_variable_declaration()
                )
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
            preprocessors=preprocessors,
        )
        if cbuffers:
            shader.cbuffers = cbuffers
        return self.finalize_shader(shader)

    def append_shader_declaration(
        self,
        declaration,
        structs,
        functions,
        global_variables,
        constants,
        cbuffers,
        stages,
    ):
        """Append a parsed shader-body declaration to the matching collection."""
        if declaration is None:
            return
        if isinstance(declaration, StructNode):
            if getattr(declaration, "is_cbuffer", False):
                cbuffers.append(declaration)
            else:
                structs.append(declaration)
        elif isinstance(declaration, EnumNode):
            structs.append(declaration)
        elif isinstance(declaration, FunctionNode):
            functions.append(declaration)
        elif isinstance(declaration, VariableNode):
            global_variables.append(declaration)
        elif isinstance(declaration, ConstantNode):
            constants.append(declaration)
        elif isinstance(declaration, StageNode):
            if hasattr(stages, "append"):
                stages.append(declaration.stage, declaration)
            else:
                stages[declaration.stage] = declaration

    def parse_shader_stage_block(self):
        """Parse a stage-qualified block into a ``StageNode``."""
        stage_type = self.current_token[1]
        stage_enum = shader_stage_from_name(stage_type) or ShaderStage.VERTEX

        self.eat(self.current_token[0])

        stage_name = None
        if self.current_token[0] == "IDENTIFIER":
            stage_name = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("LBRACE")

        local_variables = []
        local_functions = []
        local_structs = []
        local_cbuffers = []
        parsed_functions = []
        main_function = None
        execution_config = {}
        layout_qualifiers = []

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "LAYOUT" and self.is_stage_layout_qualifier():
                layout = self.parse_stage_layout_qualifier()
                layout_qualifiers.append(layout)
                execution_config.update(self.execution_config_from_layout(layout))
            elif self.current_token[0] == "LAYOUT":
                attributes = self.parse_layout_attributes()
                if self.is_layout_buffer_block_declaration():
                    struct_node, variable_node = self.parse_layout_buffer_block(
                        attributes
                    )
                    local_structs.append(struct_node)
                    local_variables.append(variable_node)
                elif self.current_token_starts_attributed_cbuffer_declaration():
                    local_cbuffers.append(self.parse_attributed_cbuffer(attributes))
                elif self.is_cbuffer_declaration():
                    local_cbuffers.append(self.parse_cbuffer_as_struct(attributes))
                elif self.is_variable_declaration():
                    self.append_parsed_nodes(
                        local_variables, self.parse_variable_declaration(attributes)
                    )
                else:
                    self.skip_unknown_token()
            elif self.current_token_starts_attributed_cbuffer_declaration():
                cbuffer = self.parse_attributed_cbuffer()
                if cbuffer:
                    local_cbuffers.append(cbuffer)
            elif self.current_token_starts_attributed_struct_declaration():
                struct_node = self.parse_attributed_struct()
                if struct_node:
                    local_structs.append(struct_node)
            elif self.current_token[0] == "STRUCT":
                struct_node = self.parse_struct()
                if struct_node:
                    local_structs.append(struct_node)
            elif self.is_cbuffer_declaration():
                cbuffer = self.parse_cbuffer_as_struct()
                if cbuffer:
                    local_cbuffers.append(cbuffer)
            elif self.is_function_declaration():
                func = self.parse_function()
                parsed_functions.append(func)
                if func.name == "main":
                    main_function = func
            elif self.is_variable_declaration():
                self.append_parsed_nodes(
                    local_variables, self.parse_variable_declaration()
                )
            else:
                self.skip_unknown_token()

        self.eat("RBRACE")

        if main_function is None:
            main_function = next(
                (
                    func
                    for func in parsed_functions
                    if self.function_has_stage_entry_attributes(func)
                ),
                None,
            )

        local_functions.extend(
            func for func in parsed_functions if func is not main_function
        )

        if not main_function:
            main_function = FunctionNode(
                name="main",
                return_type=PrimitiveType("void"),
                parameters=[],
                body=BlockNode([]),
            )

        explicit_stage_entry = self.function_has_explicit_stage_entry_attribute(
            main_function
        )
        if explicit_stage_entry:
            main_function.preserve_stage_entry_name = True
            main_function.attributes = [
                attr
                for attr in getattr(main_function, "attributes", []) or []
                if str(getattr(attr, "name", "")).lower() != "stage_entry"
            ]

        if stage_name and not explicit_stage_entry:
            main_function.name = stage_name

        execution_config.update(
            self.execution_config_from_function_attributes(main_function)
        )

        return StageNode(
            stage=stage_enum,
            entry_point=main_function,
            local_variables=local_variables,
            local_functions=local_functions,
            local_structs=local_structs,
            local_cbuffers=local_cbuffers,
            execution_config=execution_config,
            layout_qualifiers=layout_qualifiers,
        )

    def function_has_stage_entry_attributes(self, function):
        """Return whether a non-main stage function carries entry metadata."""
        stage_entry_attributes = {
            "stage_entry",
            "numthreads",
            "outputtopology",
            "max_vertices",
            "max_primitives",
            "outputcontrolpoints",
            "domain",
            "partitioning",
            "patchconstantfunc",
            "maxtessfactor",
        }
        return any(
            str(getattr(attr, "name", "")).lower() in stage_entry_attributes
            for attr in getattr(function, "attributes", []) or []
        )

    def function_has_explicit_stage_entry_attribute(self, function):
        """Return whether a function explicitly asks to preserve entry identity."""
        return any(
            str(getattr(attr, "name", "")).lower() == "stage_entry"
            for attr in getattr(function, "attributes", []) or []
        )

    def parse_import(self):
        """Parse an ``import``, ``use``, or ``from ... import`` declaration."""
        if self.current_token[0] == "FROM":
            self.eat("FROM")
            path = self.parse_import_path()
            self.eat("IMPORT")

            items = []
            while True:
                items.append(self.parse_import_path(allow_string=False))
                if self.current_token[0] != "COMMA":
                    break
                self.eat("COMMA")

            self.eat("SEMICOLON")
            return ImportNode(path=path, items=items)

        if self.current_token[0] == "IMPORT":
            self.eat("IMPORT")
        else:
            self.eat("USE")

        path = self.parse_import_path()

        alias = None
        items = None

        if self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.parse_binding_identifier()

        self.eat("SEMICOLON")

        return ImportNode(path=path, alias=alias, items=items)

    def parse_import_path(self, allow_string=True):
        if allow_string and self.current_token[0] == "STRING_LITERAL":
            value = self.current_token[1]
            self.eat("STRING_LITERAL")
            return value[1:-1]

        path = str(self.parse_binding_identifier())
        while self.current_token[0] in {"DOT", "DOUBLE_COLON"}:
            separator = "." if self.current_token[0] == "DOT" else "::"
            self.eat(self.current_token[0])
            path += separator + str(self.parse_binding_identifier())

        return path

    def parse_square_attributes(self, allow_single_square=True):
        """Parse HLSL-style ``[attr]`` or Metal-style ``[[attr]]`` metadata."""
        attributes = []

        while self.current_token_starts_square_attribute(allow_single_square):
            double_bracket = self.peek()[0] == "LBRACKET"
            self.eat("LBRACKET")
            if double_bracket:
                self.eat("LBRACKET")

            while self.current_token[0] != "RBRACKET":
                name = str(self.current_token[1])
                self.eat(self.current_token[0])

                arguments = []
                if self.current_token[0] == "LPAREN":
                    self.eat("LPAREN")
                    while self.current_token[0] != "RPAREN":
                        arguments.append(self.parse_expression())
                        if self.current_token[0] == "COMMA":
                            self.eat("COMMA")
                            continue
                        break
                    self.eat("RPAREN")

                attributes.append(AttributeNode(name=name, arguments=arguments))

                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break

            self.eat("RBRACKET")
            if double_bracket:
                self.eat("RBRACKET")

        return attributes

    def current_token_starts_square_attribute(self, allow_single_square=True):
        """Return whether the current bracketed group is metadata."""
        if self.current_token[0] != "LBRACKET":
            return False

        offset = 1
        double_bracket = self.peek()[0] == "LBRACKET"
        if double_bracket:
            offset = 2
        elif not allow_single_square:
            return False

        name_token_type, name_token_value = self.peek(offset)
        if name_token_type in {"RBRACKET", "EOF"}:
            return False
        if not isinstance(name_token_value, str) or not name_token_value.isidentifier():
            return False

        return self.peek(offset + 1)[0] in {"LPAREN", "COMMA", "RBRACKET"}

    def parse_attribute_annotations(self, allow_single_square=True):
        """Parse consecutive ``@``, HLSL ``[]``, or Metal ``[[]]`` attributes."""
        attributes = []
        while self.current_token[0] in [
            "AT",
            "ATTRIBUTE",
        ] or self.current_token_starts_square_attribute(allow_single_square):
            if self.current_token[0] in ["AT", "ATTRIBUTE"]:
                attributes.extend(self.parse_attributes())
            else:
                attributes.extend(self.parse_square_attributes(allow_single_square))
        return attributes

    def parse_post_declaration_attributes(self):
        """Parse declaration metadata that appears after a name or parameter list."""
        attributes = []
        while (
            self.current_token[0] == "COLON"
            or self.current_token[0] in ["AT", "ATTRIBUTE"]
            or self.current_token_starts_square_attribute(allow_single_square=False)
        ):
            if self.current_token[0] == "COLON":
                attributes.extend(self.parse_colon_semantic_attributes())
            else:
                attributes.extend(
                    self.parse_attribute_annotations(allow_single_square=False)
                )
        return attributes

    def parse_colon_semantic_attributes(self):
        """Parse HLSL-style ``: SEMANTIC`` metadata into attributes."""
        if self.current_token[0] != "COLON":
            return []

        self.eat("COLON")
        name = str(self.current_token[1])
        self.eat(self.current_token[0])

        arguments = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                arguments.append(self.parse_expression())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break
            self.eat("RPAREN")

        return [AttributeNode(name=name, arguments=arguments)]

    def parse_compute_layout(self):
        """Parse compute local-size layout metadata inside a stage block."""
        return self.execution_config_from_layout(self.parse_stage_layout_qualifier())

    def parse_layout_attributes(self):
        """Parse a ``layout(...)`` qualifier into attributes."""
        attributes = []
        self.eat("LAYOUT")
        self.eat("LPAREN")

        while self.current_token[0] != "RPAREN":
            name = self.current_token[1]
            self.eat(self.current_token[0])

            arguments = []
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                arguments.append(self.parse_expression())

            attributes.append(AttributeNode(name=name, arguments=arguments))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")
        return attributes

    def parse_stage_layout_qualifier(self):
        """Parse stage-level ``layout(...) in/out;`` metadata."""
        entries = []
        self.eat("LAYOUT")
        self.eat("LPAREN")

        while self.current_token[0] != "RPAREN":
            key = self.current_token[1]
            self.eat(self.current_token[0])

            arguments = []
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                arguments.append(self.parse_expression())

            entries.append(AttributeNode(name=key, arguments=arguments))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")

        direction = None
        if self.current_token[0] in {"IN", "IDENTIFIER"} and self.current_token[1] in {
            "in",
            "out",
            "patch",
        }:
            direction = self.current_token[1]
            self.eat(self.current_token[0])
        self.eat("SEMICOLON")
        return LayoutQualifierNode(entries=entries, direction=direction)

    def execution_config_from_layout(self, layout):
        """Return execution metadata implied by a stage layout qualifier."""
        if getattr(layout, "direction", None) != "in":
            return {}

        execution_config = {}
        for entry in getattr(layout, "entries", []) or []:
            if entry.name not in {"local_size_x", "local_size_y", "local_size_z"}:
                continue
            if len(entry.arguments) != 1:
                continue
            execution_config[entry.name] = self.layout_argument_to_string(
                entry.arguments[0]
            )
        return execution_config

    def execution_config_from_function_attributes(self, function):
        """Return execution metadata implied by stage function attributes."""
        for attr in getattr(function, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name not in {"local_size", "numthreads", "workgroup_size"}:
                continue

            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 3:
                continue

            values = tuple(self.layout_argument_to_string(arg) for arg in arguments)
            return {"numthreads": values}

        return {}

    def layout_argument_to_string(self, argument):
        """Return the source-like value for simple layout argument expressions."""
        if isinstance(argument, BinaryOpNode):
            left = self.layout_argument_to_string(argument.left)
            right = self.layout_argument_to_string(argument.right)
            return f"{left} {argument.op} {right}"
        if isinstance(argument, UnaryOpNode):
            operand = self.layout_argument_to_string(argument.operand)
            if argument.is_postfix:
                return f"{operand}{argument.op}"
            return f"{argument.op}{operand}"
        if hasattr(argument, "value"):
            return str(argument.value)
        if hasattr(argument, "name"):
            return str(argument.name)
        return str(argument)

    def is_stage_layout_qualifier(self):
        """Return whether the current layout is a standalone stage qualifier."""
        if self.current_token[0] != "LAYOUT" or self.peek(1)[0] != "LPAREN":
            return False

        depth = 1
        offset = 2
        while depth > 0:
            token_type, _token_value = self.peek(offset)
            if token_type == "EOF":
                return False
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
            offset += 1

        direction_type, direction_value = self.peek(offset)
        if direction_type == "IN":
            direction_value = "in"
        if direction_type != "IDENTIFIER" and direction_type != "IN":
            return False
        if direction_value not in {"in", "out", "patch"}:
            return False
        return self.peek(offset + 1)[0] == "SEMICOLON"

    def is_layout_buffer_block_declaration(self):
        """Return whether current tokens start a GLSL-style buffer block."""
        offset = 0
        while self.token_is_layout_buffer_block_qualifier(self.peek(offset)):
            offset += 1
        return (
            self.peek(offset)[0] == "BUFFER"
            and self.peek(offset + 1)[0] == "IDENTIFIER"
            and self.peek(offset + 2)[0] == "LBRACE"
        )

    def parse_layout_buffer_block(self, layout_attributes):
        """Parse ``layout(...) [readonly] buffer Block { ... } instance;``."""
        qualifiers = []
        while self.token_is_layout_buffer_block_qualifier(self.current_token):
            token_type, token_value = self.current_token
            qualifiers.append(str(token_value).lower())
            self.eat(token_type)

        self.eat("BUFFER")
        block_name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            member = self.parse_struct_member()
            self.append_parsed_nodes(members, member)
        self.eat("RBRACE")

        variable_name = self.default_layout_buffer_instance_name(block_name)
        variable_type = NamedType(block_name)
        if self.current_token[0] == "IDENTIFIER":
            variable_name = self.current_token[1]
            self.eat("IDENTIFIER")

        while self.current_token[
            0
        ] == "LBRACKET" and not self.current_token_starts_square_attribute(
            allow_single_square=False
        ):
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            variable_type = ArrayType(variable_type, size)

        self.eat("SEMICOLON")

        struct_node = StructNode(name=block_name, members=members)
        variable_node = VariableNode(
            name=variable_name,
            var_type=variable_type,
            qualifiers=qualifiers + ["buffer"],
            attributes=self.glsl_buffer_block_attributes(layout_attributes),
        )
        return struct_node, variable_node

    def glsl_buffer_block_attributes(self, layout_attributes):
        """Convert GLSL layout metadata into CrossGL buffer-block attributes."""
        attributes = []
        block_layout = self.glsl_buffer_block_layout_attribute(layout_attributes)
        if block_layout is not None:
            attributes.append(
                AttributeNode(
                    "glsl_buffer_block",
                    [IdentifierNode(block_layout)],
                )
            )
        for attr in layout_attributes:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name != block_layout:
                attributes.append(attr)
        return attributes

    def glsl_buffer_block_layout_attribute(self, layout_attributes):
        """Return the GLSL memory layout name from layout attributes."""
        for attr in layout_attributes:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def token_is_layout_buffer_block_qualifier(self, token):
        """Return whether token is a qualifier before a GLSL buffer block."""
        token_type, token_value = token
        if token_type == "BUFFER":
            return False
        return self.token_is_variable_qualifier(token_type, token_value)

    def default_layout_buffer_instance_name(self, block_name):
        """Return a stable fallback instance name for anonymous buffer blocks."""
        if not block_name:
            return "bufferBlock"
        return block_name[0].lower() + block_name[1:]

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

    def parse_attributed_cbuffer(self, leading_attributes=None):
        attributes = list(leading_attributes or [])
        attributes.extend(self.parse_attribute_annotations())
        return self.parse_cbuffer_as_struct(attributes)

    def current_token_starts_attributed_cbuffer_declaration(self):
        """Return whether leading attributes are followed by a cbuffer."""
        if not (
            self.current_token[0] in ["AT", "ATTRIBUTE"]
            or self.current_token_starts_square_attribute()
        ):
            return False

        saved_pos = self.pos
        saved_token = self.current_token
        try:
            attributes = self.parse_attribute_annotations()
            return bool(attributes and self.is_cbuffer_declaration())
        finally:
            self.pos = saved_pos
            self.current_token = saved_token

    def parse_attributed_struct(self):
        attributes = self.parse_attribute_annotations()
        return self.parse_struct(attributes)

    def current_token_starts_attributed_struct_declaration(self):
        """Return whether leading attributes are followed by ``struct``."""
        if not (
            self.current_token[0] in ["AT", "ATTRIBUTE"]
            or self.current_token_starts_square_attribute()
        ):
            return False

        saved_pos = self.pos
        saved_token = self.current_token
        try:
            attributes = self.parse_attribute_annotations()
            return bool(attributes and self.current_token[0] == "STRUCT")
        finally:
            self.pos = saved_pos
            self.current_token = saved_token

    def parse_struct(self, leading_attributes=None):
        """Parse a struct declaration and its member list."""
        if self.current_token[0] == "EOF":
            return None

        attributes = list(leading_attributes or [])

        self.eat("STRUCT")

        if self.current_token[0] != "IDENTIFIER":
            return None

        name = self.parse_binding_identifier()

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        attributes.extend(self.parse_attribute_annotations(allow_single_square=False))

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
            self.append_parsed_nodes(members, member)

        if self.current_token[0] == "EOF":
            return None

        self.eat("RBRACE")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return StructNode(
            name=name,
            members=members,
            generic_params=generic_params,
            attributes=attributes,
        )

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

        leading_attributes = []
        while True:
            if self.current_token[0] == "LAYOUT":
                leading_attributes.extend(self.parse_layout_attributes())
                continue

            attributes = self.parse_attribute_annotations()
            if attributes:
                leading_attributes.extend(attributes)
                continue

            break

        if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
            name = self.parse_binding_identifier()
            self.eat("COLON")
            member_type = self.parse_type()

            attributes = list(leading_attributes)
            attributes.extend(self.parse_post_declaration_attributes())

            if self.current_token[0] not in ["SEMICOLON", "COMMA", "RBRACE"]:
                while self.current_token[0] not in [
                    "SEMICOLON",
                    "COMMA",
                    "RBRACE",
                    "EOF",
                ]:
                    self.skip_unknown_token()
                if self.current_token[0] in ["SEMICOLON", "COMMA"]:
                    self.skip_unknown_token()
                return None

            if self.current_token[0] in ["SEMICOLON", "COMMA"]:
                self.eat(self.current_token[0])
            return StructMemberNode(
                name=name, member_type=member_type, attributes=attributes
            )

        qualifier_attributes = []
        if self.current_token_is_variable_qualifier():
            qualifier_attributes = [
                AttributeNode(name=qualifier)
                for qualifier in self.parse_variable_qualifiers()
            ]

        member_type = self.parse_type()
        member_type = self.parse_array_suffixes(member_type)

        if not self.current_token_is_binding_identifier():
            # Skip malformed member
            while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "SEMICOLON":
                self.skip_unknown_token()
            return None

        members = []
        while True:
            if not self.current_token_is_binding_identifier():
                while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                    self.skip_unknown_token()
                if self.current_token[0] == "SEMICOLON":
                    self.skip_unknown_token()
                return members or None

            name = self.parse_binding_identifier()
            declarator_type = self.parse_array_suffixes(deepcopy(member_type))

            attributes = list(leading_attributes)
            attributes.extend(qualifier_attributes)
            attributes.extend(self.parse_post_declaration_attributes())

            member = StructMemberNode(
                name=name, member_type=declarator_type, attributes=attributes
            )

            if self.current_token[0] == "COMMA":
                members.append(member)
                self.eat("COMMA")
                continue

            if self.current_token[0] == "SEMICOLON":
                members.append(member)
                self.eat("SEMICOLON")
                return members[0] if len(members) == 1 else members

            # Skip malformed member suffix while preserving earlier declarators.
            while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                self.skip_unknown_token()
            if self.current_token[0] == "SEMICOLON":
                self.skip_unknown_token()
            return members or None

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
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                elif self.peek()[0] == "COLON":
                    member_name = self.current_token[1]
                    self.eat(self.current_token[0])
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
        ):
            if self.current_token[0] in ["AT", "ATTRIBUTE"]:
                attributes.extend(self.parse_attributes())
            elif self.current_token_starts_square_attribute():
                attributes.extend(self.parse_square_attributes())
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
            return_type = self.parse_type()

        if not (
            self.current_token[0] == "KERNEL"
            or self.current_token_is_binding_identifier()
        ):
            raise SyntaxError(f"Expected function name, got {self.current_token[0]}")
        name = self.parse_binding_identifier()

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        self.eat("LPAREN")
        parameters = self.parse_parameter_list()
        self.eat("RPAREN")

        if self.is_arrow_token():
            self.eat_arrow()
            attributes.extend(self.parse_return_type_attributes())
            return_type = self.parse_type()

        post_attributes = self.parse_post_declaration_attributes()

        body = None
        if self.current_token[0] == "LBRACE":
            body_start_pos = self.pos
            body_start_token = self.current_token
            try:
                body = self.parse_block()
            except SyntaxError:
                self.pos = body_start_pos
                self.current_token = body_start_token
                self.skip_balanced_declaration()
                body = BlockNode([])
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

    def parse_return_type_attributes(self):
        """Parse WGSL-style metadata between ``->`` and the return type."""
        return self.parse_attribute_annotations(allow_single_square=False)

    def parse_parameter_list(self):
        """Parse a comma-separated function parameter list."""
        parameters = []

        if self.current_token[0] == "VOID" and self.peek()[0] == "RPAREN":
            self.eat("VOID")
            return parameters

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
        attributes.extend(self.parse_attribute_annotations())

        if self.current_token[0] == "VAR":
            return self.parse_resource_parameter(attributes)

        qualifiers = self.parse_parameter_qualifiers()

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
            name = self.parse_binding_identifier()
            self.eat("COLON")
            param_type = self.parse_type()
        elif (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "self"
            and self.peek()[0] in ["COMMA", "RPAREN"]
        ):
            name = self.parse_binding_identifier()
            param_type = NamedType("Self")
        else:
            param_type = self.parse_type()
            name = self.parse_binding_identifier()

        while self.current_token[
            0
        ] == "LBRACKET" and not self.current_token_starts_square_attribute(
            allow_single_square=False
        ):
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            param_type = ArrayType(param_type, size)

        attributes.extend(self.parse_post_declaration_attributes())

        default_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            default_value = self.parse_expression()

        return ParameterNode(
            name=name,
            param_type=param_type,
            default_value=default_value,
            attributes=attributes,
            qualifiers=qualifiers,
            is_mutable=is_mutable,
        )

    def parse_parameter_qualifiers(self):
        """Parse backend-neutral parameter qualifiers when present."""
        qualifiers = []

        while True:
            token_type, token_value = self.current_token
            normalized = str(token_value).lower() if token_value else ""

            if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
                return qualifiers

            if self.current_token_is_parameter_qualifier():
                qualifiers.append(normalized)
                self.eat(token_type)
                continue

            if (
                token_type == "IDENTIFIER"
                and normalized in PARAMETER_PRIMITIVE_QUALIFIER_NAMES
                and self.current_token_starts_qualified_parameter_type()
            ):
                qualifiers.append(normalized)
                self.eat("IDENTIFIER")
                continue

            return qualifiers

    def current_token_starts_qualified_parameter_type(self):
        """Return whether a primitive qualifier is followed by a real parameter."""
        next_token = self.peek()
        token_after_type_start = self.peek(2)
        if next_token[0] not in self.type_start_tokens():
            return False
        return token_after_type_start[0] not in {
            "COMMA",
            "RPAREN",
            "EQUALS",
            "AT",
            "ATTRIBUTE",
        }

    def parse_resource_parameter(self, attributes):
        """Parse WGSL-style resource parameters emitted by reverse backends."""
        self.eat("VAR")
        resource_qualifiers = []

        if self.current_token[0] == "LESS_THAN":
            self.eat("LESS_THAN")
            while not self.current_token_is_type_greater_than():
                if self.current_token[0] != "COMMA":
                    resource_qualifiers.append(self.current_token[1])
                    self.eat(self.current_token[0])
                    continue
                self.eat("COMMA")
            self.eat_type_greater_than()

        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("COLON")
        param_type = self.parse_resource_parameter_type()

        attributes.extend(self.parse_attribute_annotations())

        parameter = ParameterNode(
            name=name,
            param_type=param_type,
            attributes=attributes,
        )
        parameter.resource_qualifiers = resource_qualifiers
        return parameter

    def parse_resource_parameter_type(self):
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "array":
            self.eat("IDENTIFIER")
            self.eat("LESS_THAN")
            element_type = self.parse_resource_parameter_type()
            size = None
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                size = self.current_token[1]
                self.eat(self.current_token[0])
            self.eat_type_greater_than()
            return ArrayType(element_type, size)

        return self.parse_type()

    def current_token_starts_var_address_space_declaration(self):
        """Return whether the current token starts ``var<address-space>`` syntax."""
        return self.current_token[0] == "VAR" and self.peek()[0] == "LESS_THAN"

    def parse_var_address_space_qualifiers(self):
        """Parse the address-space qualifier list in ``var<...>`` declarations."""
        qualifiers = []
        self.eat("VAR")
        self.eat("LESS_THAN")

        while (
            not self.current_token_is_type_greater_than()
            and self.current_token[0] != "EOF"
        ):
            token_type, token_value = self.current_token
            if (
                token_type != "COMMA"
                and isinstance(token_value, str)
                and token_value.isidentifier()
            ):
                qualifiers.append(token_value.lower())
            self.eat(token_type)

        self.eat_type_greater_than()
        return qualifiers

    def parse_var_address_space_declaration(self, attributes):
        """Parse ``var<workgroup> name: Type`` resource/shared declarations."""
        qualifiers = self.parse_var_address_space_qualifiers()
        name = self.parse_binding_identifier()
        self.eat("COLON")
        var_type = self.parse_resource_parameter_type()
        var_type = self.parse_array_suffixes(var_type)

        attributes.extend(self.parse_post_declaration_attributes())
        var_type = self.parse_array_suffixes(var_type)

        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()

        self.eat("SEMICOLON")

        node = VariableNode(
            name=name,
            var_type=var_type,
            initial_value=initial_value,
            qualifiers=qualifiers,
            attributes=attributes,
            is_mutable="const" not in qualifiers,
        )
        node.is_var_address_space = True
        node.address_space_qualifiers = list(qualifiers)
        return node

    def parse_variable_declaration(self, leading_attributes=None):
        """Parse a variable declaration, including qualifiers and attributes."""
        attributes = list(leading_attributes or [])
        attributes.extend(self.parse_attribute_annotations())

        if self.current_token_starts_var_address_space_declaration():
            return self.parse_var_address_space_declaration(attributes)

        is_colon_style_var = (
            self.current_token[0] == "VAR"
            and self.peek()[0] != "LESS_THAN"
            and self.peek(2)[0] == "COLON"
        )

        qualifiers = [] if is_colon_style_var else self.parse_variable_qualifiers()

        if is_colon_style_var:
            self.eat("VAR")
            name = self.parse_binding_identifier()
            self.eat("COLON")
            var_type = self.parse_type()
            var_type = self.parse_array_suffixes(var_type)

            attributes.extend(self.parse_post_declaration_attributes())
            var_type = self.parse_array_suffixes(var_type)

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

        else:
            var_type = self.parse_type()
            declarations = []
            while True:
                declarations.append(
                    self.parse_variable_declarator(var_type, qualifiers, attributes)
                )
                if self.current_token[0] != "COMMA":
                    break
                self.eat("COMMA")

            self.eat("SEMICOLON")
            return declarations[0] if len(declarations) == 1 else declarations

    def parse_variable_declarator(self, base_type, qualifiers, declaration_attributes):
        """Parse one declarator after a shared C-style variable type."""
        var_type = self.parse_array_suffixes(deepcopy(base_type))
        name = self.parse_binding_identifier()
        generic_params = self.parse_declaration_square_generic_parameters()
        var_type = self.parse_array_suffixes(var_type)

        attributes = list(declaration_attributes)
        attributes.extend(self.parse_post_declaration_attributes())
        var_type = self.parse_array_suffixes(var_type)

        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()

        variable = VariableNode(
            name=name,
            var_type=var_type,
            initial_value=initial_value,
            qualifiers=list(qualifiers),
            attributes=attributes,
            is_mutable="const" not in qualifiers,
        )
        variable.generic_params = generic_params
        return variable

    def parse_declaration_square_generic_parameters(self):
        """Parse declaration-level square generic parameters after a binding name."""
        if (
            self.current_token[0] == "LBRACKET"
            and not self.current_token_starts_square_attribute(
                allow_single_square=False
            )
            and self.square_brackets_form_type_arguments()
        ):
            return self.parse_square_generic_arguments()
        return []

    def parse_array_suffixes(self, type_node):
        """Parse one or more non-attribute array suffixes after a type/name."""
        while (
            self.current_token[0] == "LBRACKET"
            and not self.current_token_starts_square_attribute(
                allow_single_square=False
            )
            and not self.square_brackets_form_type_arguments()
        ):
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            type_node = ArrayType(type_node, size)
        return type_node

    def parse_variable_qualifiers(self):
        """Parse declaration qualifiers that precede a variable type."""
        qualifiers = []
        while self.current_token_is_variable_qualifier():
            token_type, token_value = self.current_token
            qualifiers.append(str(token_value).lower())
            self.eat(token_type)
        return qualifiers

    def current_token_is_variable_qualifier(self):
        """Return whether the current token is a variable declaration qualifier."""
        token_type, token_value = self.current_token
        return self.token_is_variable_qualifier(token_type, token_value)

    def token_is_variable_qualifier(self, token_type, token_value):
        """Return whether a token is a variable declaration qualifier."""
        if token_type in VARIABLE_QUALIFIER_TOKEN_TYPES:
            return True
        return (
            token_type == "IDENTIFIER" and str(token_value) in VARIABLE_QUALIFIER_NAMES
        )

    def current_token_is_parameter_qualifier(self):
        """Return whether the current token is a parameter qualifier."""
        token_type, token_value = self.current_token
        if token_type in PARAMETER_QUALIFIER_TOKEN_TYPES:
            return True
        return (
            token_type == "IDENTIFIER" and str(token_value) in VARIABLE_QUALIFIER_NAMES
        )

    def is_variable_declaration(self):
        """
        Lookahead check for variable declarations.
        Handles complex cases and distinguishes from function calls and member access.
        """
        saved_pos = self.pos
        saved_token = self.current_token
        saved_tokens = list(self.tokens)

        try:
            self.parse_attribute_annotations()

            if (
                self.current_token[0] == "VAR"
                and self.peek()[0] != "LESS_THAN"
                and self.peek(2)[0] == "COLON"
            ):
                return True

            if self.current_token_starts_var_address_space_declaration():
                return True

            self.parse_variable_qualifiers()

            if not self.is_type_token():
                return False

            self.advance_over_type()

            if (
                self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}
                and not self.pointer_suffix_starts_declaration()
            ):
                return False

            self.advance_over_pointer_suffix()

            if not self.current_token_is_declaration_binding_identifier():
                return False

            self.parse_binding_identifier()

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
            self.tokens[:] = saved_tokens
            self.pos = saved_pos
            self.current_token = saved_token

    def pointer_suffix_starts_declaration(self):
        """Return whether pointer/reference suffix lookahead forms a declaration."""
        offset = 0
        while self.peek(offset)[0] in {"MULTIPLY", "BITWISE_AND"}:
            token_type = self.peek(offset)[0]
            offset += 1
            if token_type == "BITWISE_AND" and self.peek(offset)[0] == "MUT":
                offset += 1

        if self.peek(offset)[0] != "IDENTIFIER":
            return False

        next_token = self.peek(offset + 1)[0]
        if next_token in {"LBRACKET", "EQUALS", "SEMICOLON", "COMMA"}:
            return True
        return next_token == "RPAREN" and self.in_parameter_context()

    def advance_over_pointer_suffix(self):
        """Advance over pointer/reference suffix tokens in type lookahead."""
        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            token_type = self.current_token[0]
            self.eat(token_type)
            if token_type == "BITWISE_AND" and self.current_token[0] == "MUT":
                self.eat("MUT")

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
        name = self.parse_binding_identifier()
        const_type = self.parse_array_suffixes(const_type)
        attributes = self.parse_post_declaration_attributes()

        self.eat("EQUALS")
        value = self.parse_expression()
        self.eat("SEMICOLON")

        return ConstantNode(
            name=name, const_type=const_type, value=value, attributes=attributes
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
            sampler_types = {
                "SAMPLER": "sampler",
                "SAMPLER1D": "sampler1D",
                "SAMPLER1DARRAY": "sampler1DArray",
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
                "ISAMPLER1D": "isampler1D",
                "ISAMPLER1DARRAY": "isampler1DArray",
                "ISAMPLER2D": "isampler2D",
                "ISAMPLER3D": "isampler3D",
                "ISAMPLERCUBE": "isamplerCube",
                "ISAMPLERCUBEARRAY": "isamplerCubeArray",
                "ISAMPLER2DARRAY": "isampler2DArray",
                "ISAMPLER2DMS": "isampler2DMS",
                "ISAMPLER2DMSARRAY": "isampler2DMSArray",
                "USAMPLER1D": "usampler1D",
                "USAMPLER1DARRAY": "usampler1DArray",
                "USAMPLER2D": "usampler2D",
                "USAMPLER3D": "usampler3D",
                "USAMPLERCUBE": "usamplerCube",
                "USAMPLERCUBEARRAY": "usamplerCubeArray",
                "USAMPLER2DARRAY": "usampler2DArray",
                "USAMPLER2DMS": "usampler2DMS",
                "USAMPLER2DMSARRAY": "usampler2DMSArray",
                "IIMAGE1D": "iimage1D",
                "IIMAGE1DARRAY": "iimage1DArray",
                "IIMAGE2D": "iimage2D",
                "IIMAGE3D": "iimage3D",
                "IIMAGECUBE": "iimageCube",
                "IIMAGECUBEARRAY": "iimageCubeArray",
                "IIMAGE2DARRAY": "iimage2DArray",
                "IIMAGE2DMS": "iimage2DMS",
                "IIMAGE2DMSARRAY": "iimage2DMSArray",
                "UIMAGE1D": "uimage1D",
                "UIMAGE1DARRAY": "uimage1DArray",
                "UIMAGE2D": "uimage2D",
                "UIMAGE3D": "uimage3D",
                "UIMAGECUBE": "uimageCube",
                "UIMAGECUBEARRAY": "uimageCubeArray",
                "UIMAGE2DARRAY": "uimage2DArray",
                "UIMAGE2DMS": "uimage2DMS",
                "UIMAGE2DMSARRAY": "uimage2DMSArray",
                "IMAGE1D": "image1D",
                "IMAGE1DARRAY": "image1DArray",
                "IMAGE2D": "image2D",
                "IMAGE3D": "image3D",
                "IMAGECUBE": "imageCube",
                "IMAGECUBEARRAY": "imageCubeArray",
                "IMAGE2DARRAY": "image2DArray",
                "IMAGE2DMS": "image2DMS",
                "IMAGE2DMSARRAY": "image2DMSArray",
            }
            token_type = self.current_token[0]
            self.eat(token_type)
            base_type = NamedType(sampler_types[token_type])

        elif self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "def":
            base_type = self.parse_callable_type()

        elif self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            while self.current_token[0] in {"DOT", "DOUBLE_COLON"}:
                separator = self.current_token[1]
                self.eat(self.current_token[0])
                segment = self.parse_binding_identifier()
                name = f"{name}{separator}{segment}"

            generic_args = []
            if self.current_token[0] == "LESS_THAN":
                generic_args = self.parse_generic_arguments()
            elif (
                self.current_token[0] == "LBRACKET"
                and self.square_brackets_form_type_arguments()
            ):
                generic_args = self.parse_square_generic_arguments()

            while self.current_token[0] == "LPAREN":
                name = f"{name}({self.parse_balanced_parenthesized_token_text()})"

            while self.current_token[0] in {"DOT", "DOUBLE_COLON"}:
                separator = self.current_token[1]
                self.eat(self.current_token[0])
                segment = self.parse_binding_identifier()

                segment_generic_args = []
                if self.current_token[0] == "LESS_THAN":
                    segment_generic_args = self.parse_generic_arguments()
                elif self.current_token[0] == "LBRACKET" and (
                    self.square_brackets_form_type_arguments()
                    or (
                        generic_args
                        and self.square_brackets_form_member_type_arguments()
                    )
                ):
                    segment_generic_args = self.parse_square_generic_arguments()

                if generic_args:
                    name = f"{name}_{segment}"
                    generic_args.extend(segment_generic_args)
                else:
                    name = f"{name}{separator}{segment}"
                    generic_args = segment_generic_args

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
            if (
                isinstance(base_type, NamedType)
                and self.square_brackets_form_type_arguments()
            ):
                base_type.generic_args.extend(self.parse_square_generic_arguments())
                continue
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            base_type = ArrayType(base_type, size)

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
            elif depth == 1 and token_type in {"COMMA", "COLON", "EQUALS"}:
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
            args.append(self.parse_square_generic_argument())

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

    def parse_generic_arguments(self):
        """Parse generic type arguments after ``<``."""
        self.eat("LESS_THAN")
        args = []

        while not self.current_token_is_type_greater_than():
            if self.current_token_starts_qualified_identifier():
                args.append(IdentifierNode(self.parse_qualified_identifier()))
            elif self.current_token_starts_type():
                args.append(self.parse_type())
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
                args.append(self.parse_literal())
            else:
                args.append(self.parse_expression())

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat_type_greater_than()
        return args

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
            return str(value).lower() if isinstance(value, bool) else str(value)
        if isinstance(type_node, ArrayAccessNode):
            array = self.format_type_argument(type_node.array)
            index = self.format_type_argument(type_node.index)
            return f"{array}[{index}]"
        if isinstance(type_node, BinaryOpNode):
            left = self.format_type_argument(type_node.left)
            right = self.format_type_argument(type_node.right)
            return f"{left} {type_node.operator} {right}"
        if isinstance(type_node, FunctionCallNode):
            function = self.format_type_argument(type_node.function)
            args = ", ".join(
                self.format_type_argument(argument) for argument in type_node.arguments
            )
            return f"{function}({args})"
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
            return f"{self.format_expression_path(type_node.object_expr)}.{type_node.member}"
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
            if name in {"gl_FragData"} and self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] != "RBRACKET":
                    arguments.append(self.parse_expression())
                self.eat("RBRACKET")
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
            self.append_parsed_nodes(statements, stmt)

        self.eat("RBRACE")
        return BlockNode(statements)

    def parse_statement(self):
        """Parse any statement form supported by CrossGL."""
        if self.current_token_starts_statement_attribute():
            return self.parse_attributed_statement()
        if self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
        elif self.current_token[0] == "DO":
            return self.parse_do_while_statement()
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
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return ExpressionStatementNode(expr)
            if self.current_token[0] == "RBRACE":
                return ExpressionStatementNode(expr, is_tail_expression=True)
            self.eat("SEMICOLON")
            return ExpressionStatementNode(expr)

    def current_token_starts_statement_attribute(self):
        """Return whether the current token starts statement-level metadata."""
        return (
            self.current_token[0] in {"AT", "ATTRIBUTE"}
            or self.current_token_starts_square_attribute()
        )

    def parse_attributed_statement(self):
        """Parse metadata that decorates the following statement."""
        attributes = self.parse_attribute_annotations()
        statement = self.parse_statement()
        if isinstance(statement, list):
            for node in statement:
                existing = getattr(node, "attributes", [])
                node.attributes = [*attributes, *existing]
            return statement
        existing = getattr(statement, "attributes", [])
        statement.attributes = [*attributes, *existing]
        return statement

    def parse_let_declaration(self):
        """Parse Rust-style ``let [mut] name [: type] = expr;`` declarations."""
        self.eat("LET")

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        name = self.parse_binding_identifier()
        generic_params = self.parse_declaration_square_generic_parameters()

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
        var_node.generic_params = generic_params

        if is_mutable:
            var_node.qualifiers = getattr(var_node, "qualifiers", []) + ["mut"]

        var_node.is_let_declaration = True

        return var_node

    def parse_binding_identifier(self):
        """Parse an identifier in binding position, including keyword-like names."""
        token_type, token_value = self.current_token
        if not self.current_token_is_binding_identifier():
            raise SyntaxError(f"Expected binding name, got {token_type}")

        self.eat(token_type)
        return token_value

    def current_token_is_binding_identifier(self):
        token_value = self.current_token[1]
        return isinstance(token_value, str) and token_value.isidentifier()

    def current_token_is_declaration_binding_identifier(self):
        token_type, token_value = self.current_token
        if token_type == "IDENTIFIER":
            return True
        if not (isinstance(token_value, str) and token_value.isidentifier()):
            return False
        return self.peek()[0] in {
            "AT",
            "ATTRIBUTE",
            "COMMA",
            "EQUALS",
            "LBRACKET",
            "RPAREN",
            "SEMICOLON",
        }

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
            update = self.parse_expression_sequence()

        self.eat("RPAREN")
        body = self.parse_statement()

        return ForNode(init=init, condition=condition, update=update, body=body)

    def parse_expression_sequence(self):
        expressions = [self.parse_expression()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            expressions.append(self.parse_expression())
        return expressions[0] if len(expressions) == 1 else expressions

    def parse_for_in_statement_after_for(self):
        """Parse a ``for pattern in iterable`` loop after ``for`` is consumed."""
        pattern = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("IN")
        previous_suppression = getattr(self, "suppress_braced_constructor", False)
        self.suppress_braced_constructor = True
        try:
            iterable = self.parse_expression()
        finally:
            self.suppress_braced_constructor = previous_suppression
        body = self.parse_statement()

        return ForInNode(pattern=pattern, iterable=iterable, body=body)

    def parse_for_loop_variable_declaration(self):
        """Parse variable declarations in for loops (without consuming semicolon)."""
        if (
            self.current_token[0] == "VAR"
            and self.peek()[0] != "LESS_THAN"
            and self.peek(2)[0] == "COLON"
        ):
            self.eat("VAR")
            name = self.parse_binding_identifier()
            self.eat("COLON")
            var_type = self.parse_type()
            var_type = self.parse_array_suffixes(var_type)
            attributes = self.parse_post_declaration_attributes()
            var_type = self.parse_array_suffixes(var_type)

            initial_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                initial_value = self.parse_expression()

            return VariableNode(
                name=name,
                var_type=var_type,
                initial_value=initial_value,
                attributes=attributes,
            )

        qualifiers = self.parse_variable_qualifiers()
        var_type = self.parse_type()
        declarations = []
        while True:
            declarations.append(
                self.parse_variable_declarator(var_type, qualifiers, [])
            )
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        # Don't consume semicolon - that's handled by the for loop parser.
        return declarations[0] if len(declarations) == 1 else declarations

    def parse_while_statement(self):
        """Parse a while loop statement."""
        self.eat("WHILE")
        condition = self.parse_expression()
        body = self.parse_statement()

        return WhileNode(condition=condition, body=body)

    def parse_do_while_statement(self):
        """Parse a do-while loop statement."""
        self.eat("DO")
        body = self.parse_statement()
        self.eat("WHILE")
        condition = self.parse_expression()
        self.eat("SEMICOLON")

        return DoWhileNode(body=body, condition=condition)

    def parse_loop_statement(self):
        """Parse an unconditional loop statement."""
        self.eat("LOOP")
        body = self.parse_statement()

        return LoopNode(body=body)

    def parse_match_statement(self):
        """Parse a match statement and all of its arms."""
        self.eat("MATCH")
        previous_suppression = getattr(self, "suppress_braced_constructor", False)
        self.suppress_braced_constructor = True
        try:
            expression = self.parse_expression()
        finally:
            self.suppress_braced_constructor = previous_suppression

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
        body = self.parse_match_arm_body()

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")

        return MatchArmNode(pattern=pattern, guard=guard, body=body)

    def parse_match_arm_body(self):
        """Parse a match arm body, including expression arms without semicolons."""
        if self.current_token[0] == "LBRACE":
            return self.parse_block()

        if self.current_token[0] in [
            "IF",
            "FOR",
            "WHILE",
            "DO",
            "LOOP",
            "MATCH",
            "SWITCH",
            "RETURN",
            "BREAK",
            "CONTINUE",
            "LET",
        ]:
            return self.parse_statement()

        expr = self.parse_expression()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ExpressionStatementNode(expr)
        return ExpressionStatementNode(expr, is_tail_expression=True)

    def parse_pattern(self):
        """Parse a match pattern."""
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "_":
            self.eat("IDENTIFIER")
            return WildcardPatternNode()

        if self.current_token[0] == "RANGE":
            self.eat("RANGE")
            return IdentifierPatternNode("..")

        if self.is_pattern_literal_token(self.current_token[0]):
            literal = self.parse_literal()
            return LiteralPatternNode(literal)

        if self.is_pattern_path_token(self.current_token[0]):
            path = self.parse_pattern_path()
            if self.current_token[0] == "LPAREN":
                return self.parse_constructor_pattern(path)
            if self.current_token[0] == "LBRACE":
                return self.parse_struct_pattern(path)
            return IdentifierPatternNode(path)

        literal = self.parse_literal()
        return LiteralPatternNode(literal)

    def is_pattern_literal_token(self, token_type):
        return token_type in [
            "BOOLEAN_LITERAL",
            "NUMBER",
            "FLOAT_NUMBER",
            "HEX_NUMBER",
            "BIN_NUMBER",
            "OCT_NUMBER",
            "STRING_LITERAL",
            "CHAR_LITERAL",
        ]

    def is_pattern_path_token(self, token_type):
        return token_type in [
            "IDENTIFIER",
            "VEC2",
            "VEC3",
            "VEC4",
            "MAT2",
            "MAT3",
            "MAT4",
            "BOOL",
            "INT",
            "UINT",
            "FLOAT",
            "DOUBLE",
        ]

    def parse_pattern_path(self):
        segments = [str(self.current_token[1])]
        self.eat(self.current_token[0])

        while self.current_token[0] == "DOUBLE_COLON":
            self.eat("DOUBLE_COLON")
            segments.append(str(self.current_token[1]))
            self.eat(self.current_token[0])

        return "::".join(segments)

    def parse_constructor_pattern(self, type_name):
        self.eat("LPAREN")
        arguments = []

        while self.current_token[0] != "RPAREN":
            arguments.append(self.parse_pattern())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        self.eat("RPAREN")
        return ConstructorPatternNode(type_name, arguments)

    def parse_struct_pattern(self, type_name):
        self.eat("LBRACE")
        field_patterns = {}
        has_rest = False

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            if self.current_token[0] == "RANGE":
                self.eat("RANGE")
                has_rest = True
                continue

            field_name = str(self.current_token[1])
            self.eat(self.current_token[0])

            if self.current_token[0] == "COLON":
                self.eat("COLON")
                field_patterns[field_name] = self.parse_pattern()
            else:
                field_patterns[field_name] = IdentifierPatternNode(field_name)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RBRACE")
        return StructPatternNode(type_name, field_patterns, has_rest=has_rest)

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
        left = self.parse_power_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_power_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_power_expression(self):
        """Parse exponentiation syntax into the canonical ``pow`` intrinsic."""
        left = self.parse_unary_expression()

        if self.current_token[0] == "POWER":
            self.eat("POWER")
            right = self.parse_power_expression()
            return FunctionCallNode(IdentifierNode("pow"), [left, right])

        return left

    def parse_unary_expression(self):
        """Parse prefix unary expressions."""
        if self.current_token[0] in [
            "NOT",
            "MINUS",
            "PLUS",
            "BITWISE_NOT",
            "BITWISE_AND",
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
                member = self.parse_binding_identifier()
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
                and isinstance(left, IdentifierNode)
                and (
                    left.name in {"vec2", "vec3", "vec4"}
                    or self.generic_suffix_is_expression_name_suffix()
                )
            ):
                generic_args = self.parse_generic_arguments()
                args = ", ".join(self.format_type_argument(arg) for arg in generic_args)
                left = IdentifierNode(f"{left.name}<{args}>")
            elif self.current_token[0] == "LPAREN":
                if isinstance(left, IdentifierNode) and left.name == "lambda":
                    arguments = self.parse_lambda_call_arguments()
                else:
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

        index = self.pos
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            in_nested_expression = paren_depth or bracket_depth or brace_depth

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth:
                bracket_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth:
                brace_depth -= 1
            elif token_type == "LESS_THAN" and not in_nested_expression:
                angle_depth += 1
            elif token_type == "GREATER_THAN" and not in_nested_expression:
                angle_depth -= 1
                if angle_depth == 0:
                    index += 1
                    while index < len(self.tokens) and self.tokens[index][0] in {
                        "COMMENT_SINGLE",
                        "COMMENT_MULTI",
                    }:
                        index += 1
                    return index < len(self.tokens) and self.tokens[index][0] in {
                        "COMMA",
                        "DOT",
                        "LBRACKET",
                        "LPAREN",
                        "RBRACE",
                        "RBRACKET",
                        "RPAREN",
                        "SEMICOLON",
                    }
            elif token_type == "BITWISE_SHIFT_RIGHT" and not in_nested_expression:
                angle_depth -= 2
                if angle_depth == 0:
                    index += 1
                    while index < len(self.tokens) and self.tokens[index][0] in {
                        "COMMENT_SINGLE",
                        "COMMENT_MULTI",
                    }:
                        index += 1
                    return index < len(self.tokens) and self.tokens[index][0] in {
                        "COMMA",
                        "DOT",
                        "LBRACKET",
                        "LPAREN",
                        "RBRACE",
                        "RBRACKET",
                        "RPAREN",
                        "SEMICOLON",
                    }

            index += 1

        return False

    def format_expression_path(self, expression):
        if isinstance(expression, IdentifierNode):
            return expression.name
        if isinstance(expression, MemberAccessNode):
            return f"{self.format_expression_path(expression.object_expr)}.{expression.member}"
        return str(expression)

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
        return ConstructorNode(NamedType(type_name), arguments, named_arguments)

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
                elements = [expr]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    if self.current_token[0] == "RPAREN":
                        break
                    elements.append(self.parse_expression())
                self.eat("RPAREN")
                return ArrayLiteralNode(elements)
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
            literal_text = value.rstrip("fF")
            literal_value = (
                float.fromhex(literal_text)
                if literal_text.lower().startswith("0x")
                else float(literal_text)
            )
            return LiteralNode(literal_value, PrimitiveType("float"))

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

    def parse_cbuffer_as_struct(self, leading_attributes=None):
        """Parse a constant/uniform buffer declaration as a struct node."""
        attributes = list(leading_attributes or [])

        if self.current_token[0] == "CBUFFER":
            self.eat("CBUFFER")
        else:
            self.eat("UNIFORM")

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        attributes.extend(self.parse_post_declaration_attributes())

        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            member = self.parse_struct_member()
            self.append_parsed_nodes(members, member)

        self.eat("RBRACE")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        node = StructNode(name=name, members=members, attributes=attributes)
        node.is_cbuffer = True
        return node

    # Helper methods
    def is_cbuffer_declaration(self):
        """Return whether the current token begins a cbuffer declaration."""
        if self.current_token[0] == "CBUFFER":
            return True
        if self.current_token[0] != "UNIFORM" or self.peek()[0] != "IDENTIFIER":
            return False

        offset = 2
        while self.peek(offset)[0] in ["AT", "ATTRIBUTE"]:
            if self.peek(offset)[0] == "ATTRIBUTE":
                offset += 1
            else:
                if self.peek(offset + 1)[0] != "IDENTIFIER":
                    return False
                offset += 2

            if self.peek(offset)[0] == "LPAREN":
                depth = 1
                offset += 1
                while depth and self.peek(offset)[0] != "EOF":
                    if self.peek(offset)[0] == "LPAREN":
                        depth += 1
                    elif self.peek(offset)[0] == "RPAREN":
                        depth -= 1
                    offset += 1

        return self.peek(offset)[0] == "LBRACE"

    def is_generic_declaration(self):
        """Return whether the current token begins a legacy generic wrapper."""
        return (
            self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "generic"
        )

    def current_token_starts_bare_function_name(self):
        """Return whether the current token is a ``fn name`` style function name."""
        if not (
            self.current_token[0] == "KERNEL"
            or self.current_token_is_binding_identifier()
        ):
            return False

        offset = 1
        if self.peek(offset)[0] == "LESS_THAN":
            depth = 1
            offset += 1
            while depth > 0 and self.peek(offset)[0] != "EOF":
                if self.peek(offset)[0] == "LESS_THAN":
                    depth += 1
                elif self.peek(offset)[0] == "GREATER_THAN":
                    depth -= 1
                elif self.peek(offset)[0] == "BITWISE_SHIFT_RIGHT":
                    depth -= 2
                offset += 1

        return self.peek(offset)[0] == "LPAREN"

    def is_arrow_token(self):
        """Return whether the current token represents ``->``."""
        return self.current_token[0] == "ARROW" or (
            self.current_token[0] == "MINUS" and self.peek()[0] == "GREATER_THAN"
        )

    def eat_arrow(self):
        """Consume ``->`` whether the lexer emitted it as one token or two."""
        if self.current_token[0] == "ARROW":
            self.eat("ARROW")
        else:
            self.eat("MINUS")
            self.eat("GREATER_THAN")

    def is_function_declaration(self):
        """Return whether the current token sequence looks like a function."""
        saved_pos = self.pos
        saved_token = self.current_token
        saved_tokens = list(self.tokens)

        try:
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
            ):
                if self.current_token[0] in ["AT", "ATTRIBUTE"]:
                    self.parse_attributes()
                elif self.current_token_starts_square_attribute():
                    self.parse_square_attributes()
                else:
                    self.eat(self.current_token[0])

            if self.current_token[0] == "FUNCTION":
                self.eat("FUNCTION")
                if self.current_token_starts_bare_function_name():
                    return True

            if self.is_type_token():
                self.advance_over_type()
                if (
                    self.current_token[0] == "KERNEL"
                    or self.current_token_is_binding_identifier()
                ):
                    self.parse_binding_identifier()
                    # Skip generic parameters if present
                    if self.current_token[0] == "LESS_THAN":
                        depth = 1
                        self.eat("LESS_THAN")
                        while depth > 0 and self.current_token[0] != "EOF":
                            if self.current_token[0] == "LESS_THAN":
                                depth += 1
                                self.eat("LESS_THAN")
                                continue
                            if self.current_token_is_type_greater_than():
                                depth -= 1
                                self.eat_type_greater_than()
                                continue
                            self.eat(self.current_token[0])

                    if self.current_token[0] == "LPAREN":
                        return True
        except:
            pass
        finally:
            self.tokens[:] = saved_tokens
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
            "IDENTIFIER",
        ]

    def advance_over_type(self):
        """Advance over a type expression during lookahead checks."""
        if self.is_type_token():
            self.eat(self.current_token[0])

            while self.current_token[0] in {"DOT", "DOUBLE_COLON"}:
                self.eat(self.current_token[0])
                if self.current_token_is_binding_identifier():
                    self.eat(self.current_token[0])
                else:
                    break

            if self.current_token[0] == "LESS_THAN":
                depth = 1
                self.eat("LESS_THAN")
                while depth > 0 and self.current_token[0] != "EOF":
                    if self.current_token[0] == "LESS_THAN":
                        depth += 1
                        self.eat("LESS_THAN")
                        continue
                    if self.current_token_is_type_greater_than():
                        depth -= 1
                        self.eat_type_greater_than()
                        continue
                    self.eat(self.current_token[0])

            while self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                bracket_depth = 1
                while bracket_depth > 0 and self.current_token[0] != "EOF":
                    if self.current_token[0] == "LBRACKET":
                        bracket_depth += 1
                    elif self.current_token[0] == "RBRACKET":
                        bracket_depth -= 1
                    self.eat(self.current_token[0])

            while self.current_token[0] == "LPAREN":
                self.parse_balanced_parenthesized_token_text()

    def parse_balanced_parenthesized_token_text(self):
        """Consume a parenthesized token payload and return compact source text."""
        self.eat("LPAREN")
        payload_tokens = []
        depth = 1

        while depth and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token

            if token_type == "LPAREN":
                depth += 1
                payload_tokens.append((token_type, token_value))
                self.eat("LPAREN")
                continue

            if token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
                payload_tokens.append((token_type, token_value))
                self.eat("RPAREN")
                continue

            payload_tokens.append((token_type, token_value))
            self.eat(token_type)

        return self.format_raw_lambda_tokens(payload_tokens)

    def skip_unknown_token(self):
        """Consume one token when recovering from unsupported syntax."""
        if self.current_token[0] != "EOF":
            self.eat(self.current_token[0])

    def skip_balanced_declaration(self):
        """Recover past one semicolon or braced declaration without leaking tokens."""
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]

            if token_type == "RBRACE" and brace_depth == 0:
                break

            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                self.eat("SEMICOLON")
                break

            if token_type == "LBRACE":
                brace_depth += 1
                self.eat("LBRACE")
                continue

            if token_type == "RBRACE":
                self.eat("RBRACE")
                brace_depth -= 1
                if brace_depth == 0 and paren_depth == 0 and bracket_depth == 0:
                    break
                continue

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1

            self.eat(token_type)

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
        """Parse one top-level declaration or recover past unsupported input."""
        if self.current_token[0] == "EOF":
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

        if self.current_token[0] == "CONST":
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
            self.skip_unknown_token()
            return None

        if self.current_token_starts_shader_stage_block():
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

        start_pos = self.pos
        start_token = self.current_token

        try:
            return self.parse_generic_declaration_inner()
        except SyntaxError:
            self.pos = start_pos
            self.current_token = start_token
            self.skip_balanced_declaration()
            return None

    def parse_generic_declaration_inner(self):
        """Parse a generic wrapper after recovery state has been captured."""
        self.eat("IDENTIFIER")

        if self.current_token[0] != "LESS_THAN":
            return None

        generic_params = self.parse_generic_parameters()

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
        elif self.current_token[0] == "TRAIT":
            trait_node = self.parse_trait()
            if trait_node:
                trait_node.generic_params = generic_params
            return trait_node
        else:
            self.skip_balanced_declaration()
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
            generic_params = self.parse_generic_parameters()

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

        # Traits are represented as marked structs to avoid a broader AST migration.
        trait_node = StructNode(
            name=name, members=methods, attributes=[], generic_params=generic_params
        )
        trait_node.is_trait = True
        return trait_node

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
            self.append_parsed_nodes(statements, stmt)

        return CaseNode(value=value, statements=statements)

    def parse_default_case(self):
        """Parse the default switch case."""
        self.eat("DEFAULT")
        self.eat("COLON")

        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            self.append_parsed_nodes(statements, stmt)

        return CaseNode(value=None, statements=statements)

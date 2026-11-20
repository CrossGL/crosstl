"""Program structure in the CrossGL parser: programs, shader declarations, stage blocks, stage entry attributes, imports and legacy shader stages."""

from ..ast import (
    BlockNode,
    ConstantNode,
    EnumNode,
    ExecutionModel,
    FunctionNode,
    ImportNode,
    PrimitiveType,
    ShaderNode,
    ShaderStage,
    StageMap,
    StageNode,
    StructNode,
    VariableNode,
    create_legacy_shader_node,
)
from ..host_workgroup_size import (
    HOST_WORKGROUP_SIZE_ATTRIBUTE,
    HOST_WORKGROUP_SIZE_KEY,
    host_workgroup_size_mark,
)
from ..stage_utils import shader_stage_from_name
from .vocabulary import SHADER_STAGE_TOKEN_TYPES


class ParserProgramStructureMixin:
    """Program structure members of ``Parser``."""

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
            if self.parse_scope_static_assertion():
                continue
            if self.current_token[0] in ["IMPORT", "USE", "FROM"]:
                imports.append(self.parse_import())
            elif self.current_token[0] == "PREPROCESSOR":
                preprocessors.append(self.parse_preprocessor_directive())
            elif self.current_token[0] == "PRECISION":
                preprocessors.append(self.parse_precision_statement())
            elif self.current_token[0] == "SHADER":
                shader_node = self.parse_shader_declaration()
            elif self.current_token_starts_attributed_cbuffer_declaration():
                self.append_constant_block(
                    self.parse_attributed_cbuffer(), cbuffers, structs, global_variables
                )
            elif self.current_token_starts_attributed_struct_declaration():
                structs.append(self.parse_attributed_struct())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "ENUM":
                enums.append(self.parse_enum())
            elif self.is_cbuffer_declaration():
                self.append_constant_block(
                    self.parse_cbuffer_as_struct(), cbuffers, structs, global_variables
                )
            elif (
                self.current_token[0] == "CONST" and not self.is_function_declaration()
            ):
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
                    self.append_constant_block(
                        self.parse_attributed_cbuffer(attributes),
                        cbuffers,
                        structs,
                        global_variables,
                    )
                elif self.is_cbuffer_declaration():
                    self.append_constant_block(
                        self.parse_cbuffer_as_struct(attributes),
                        cbuffers,
                        structs,
                        global_variables,
                    )
                elif self.is_variable_declaration():
                    self.append_parsed_nodes(
                        global_variables, self.parse_variable_declaration(attributes)
                    )
                else:
                    self.reject_unexpected_token("a layout-qualified declaration")
            elif self.is_function_declaration():
                functions.append(self.parse_function())
            elif self.is_variable_declaration():
                self.append_parsed_nodes(
                    global_variables, self.parse_variable_declaration()
                )
            elif self.current_token[0] in SHADER_STAGE_TOKEN_TYPES:
                stage_func = self.parse_shader_stage()
                functions.append(stage_func)
            elif self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            else:
                self.reject_unexpected_token("a top-level declaration")

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
            self.reject_end_of_input(f"shader declaration {name!r}")
            if self.parse_scope_static_assertion():
                continue
            if self.current_token[0] == "PREPROCESSOR":
                preprocessors.append(self.parse_preprocessor_directive())
            elif self.current_token[0] == "PRECISION":
                preprocessors.append(self.parse_precision_statement())
            elif self.current_token_starts_shader_stage_block():
                stage_node = self.parse_shader_stage_block()
                stages.append(stage_node.stage, stage_node)
            elif self.current_token_starts_attributed_cbuffer_declaration():
                self.append_constant_block(
                    self.parse_attributed_cbuffer(), cbuffers, structs, global_variables
                )
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
                self.append_constant_block(
                    self.parse_cbuffer_as_struct(), cbuffers, structs, global_variables
                )
            elif (
                self.current_token[0] == "CONST" and not self.is_function_declaration()
            ):
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
                    self.append_constant_block(
                        self.parse_attributed_cbuffer(attributes),
                        cbuffers,
                        structs,
                        global_variables,
                    )
                elif self.is_cbuffer_declaration():
                    self.append_constant_block(
                        self.parse_cbuffer_as_struct(attributes),
                        cbuffers,
                        structs,
                        global_variables,
                    )
                elif self.is_variable_declaration():
                    self.append_parsed_nodes(
                        global_variables, self.parse_variable_declaration(attributes)
                    )
                else:
                    self.reject_unexpected_token("a layout-qualified declaration")
            elif self.is_function_declaration():
                func = self.parse_function()
                functions.append(func)
            elif self.is_variable_declaration():
                self.append_parsed_nodes(
                    global_variables, self.parse_variable_declaration()
                )
            elif self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            else:
                self.reject_unexpected_token("a shader declaration")

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
            self.reject_end_of_input(f"{stage_type} stage block")
            if self.parse_scope_static_assertion():
                continue
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
                    self.append_constant_block(
                        self.parse_attributed_cbuffer(attributes),
                        local_cbuffers,
                        local_structs,
                        local_variables,
                    )
                elif self.is_cbuffer_declaration():
                    self.append_constant_block(
                        self.parse_cbuffer_as_struct(attributes),
                        local_cbuffers,
                        local_structs,
                        local_variables,
                    )
                elif self.is_variable_declaration():
                    self.append_parsed_nodes(
                        local_variables, self.parse_variable_declaration(attributes)
                    )
                else:
                    self.reject_unexpected_token("a layout-qualified declaration")
            elif self.current_token_starts_attributed_cbuffer_declaration():
                self.append_constant_block(
                    self.parse_attributed_cbuffer(),
                    local_cbuffers,
                    local_structs,
                    local_variables,
                )
            elif self.current_token_starts_attributed_struct_declaration():
                struct_node = self.parse_attributed_struct()
                if struct_node:
                    local_structs.append(struct_node)
            elif self.current_token[0] == "STRUCT":
                struct_node = self.parse_struct()
                if struct_node:
                    local_structs.append(struct_node)
            elif self.is_cbuffer_declaration():
                self.append_constant_block(
                    self.parse_cbuffer_as_struct(),
                    local_cbuffers,
                    local_structs,
                    local_variables,
                )
            elif self.is_function_declaration():
                func = self.parse_function()
                parsed_functions.append(func)
                if func.name == "main":
                    main_function = func
            elif self.is_variable_declaration():
                self.append_parsed_nodes(
                    local_variables, self.parse_variable_declaration()
                )
            elif self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            else:
                self.reject_unexpected_token(f"a {stage_type} stage block")

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
        host_size_mark = host_workgroup_size_mark(
            getattr(main_function, "attributes", []) or [],
            self.layout_argument_to_string,
        )
        if host_size_mark is not None:
            main_function.attributes = [
                attr
                for attr in main_function.attributes
                if str(getattr(attr, "name", "")).lower()
                != HOST_WORKGROUP_SIZE_ATTRIBUTE
            ]
            execution_config[HOST_WORKGROUP_SIZE_KEY] = host_size_mark

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
            self.reject_end_of_input(f"{stage_type} stage block")
            if self.is_function_declaration():
                func = self.parse_function()
                body.append(func)
            elif self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            else:
                self.reject_unexpected_token(f"a {stage_type} stage block")

        self.eat("RBRACE")

        return FunctionNode(
            name=f"{stage_type}_main",
            return_type=PrimitiveType("void"),
            parameters=[],
            body=BlockNode(body),
            qualifiers=[stage_type],
        )

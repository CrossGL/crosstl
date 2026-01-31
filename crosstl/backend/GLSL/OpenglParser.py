from .OpenglAst import (
    ShaderNode,
    VariableNode,
    AssignmentNode,
    FunctionNode,
    ArrayAccessNode,
    BinaryOpNode,
    UnaryOpNode,
    ReturnNode,
    FunctionCallNode,
    IfNode,
    ForNode,
    WhileNode,
    DoWhileNode,
    MemberAccessNode,
    TernaryOpNode,
    StructNode,
    SwitchNode,
    CaseNode,
    BlockNode,
    NumberNode,
    PostfixOpNode,
    BreakNode,
    ContinueNode,
    DiscardNode,
)

TYPE_TOKENS = {
    "VOID",
    "BOOL",
    "INT",
    "UINT",
    "FLOAT",
    "DOUBLE",
    "VECTOR",
    "MATRIX",
    "SAMPLER2D",
    "SAMPLER3D",
    "SAMPLERCUBE",
    "SAMPLER1D",
    "SAMPLER1DARRAY",
    "SAMPLER1DSHADOW",
    "SAMPLER1DARRAYSHADOW",
    "SAMPLER2DARRAY",
    "SAMPLER2DARRAYSHADOW",
    "SAMPLERCUBEARRAY",
    "SAMPLERCUBEARRAYSHADOW",
    "SAMPLER2DSHADOW",
    "SAMPLER2DRECT",
    "SAMPLER2DRECTSHADOW",
    "SAMPLERBUFFER",
    "SAMPLERCUBESHADOW",
    "SAMPLER2DMS",
    "SAMPLER2DMSARRAY",
    "ISAMPLER1D",
    "ISAMPLER2D",
    "ISAMPLER3D",
    "ISAMPLERCUBE",
    "ISAMPLER1DARRAY",
    "ISAMPLER2DARRAY",
    "ISAMPLERCUBEARRAY",
    "ISAMPLER2DRECT",
    "ISAMPLERBUFFER",
    "ISAMPLER2DMS",
    "ISAMPLER2DMSARRAY",
    "USAMPLER1D",
    "USAMPLER2D",
    "USAMPLER3D",
    "USAMPLERCUBE",
    "USAMPLER1DARRAY",
    "USAMPLER2DARRAY",
    "USAMPLERCUBEARRAY",
    "USAMPLER2DRECT",
    "USAMPLERBUFFER",
    "USAMPLER2DMS",
    "USAMPLER2DMSARRAY",
    "IMAGE1D",
    "IMAGE2D",
    "IMAGE3D",
    "IMAGECUBE",
    "IMAGE1DARRAY",
    "IMAGE2DARRAY",
    "IMAGECUBEARRAY",
    "IMAGE2DRECT",
    "IMAGEBUFFER",
    "IMAGE2DMS",
    "IMAGE2DMSARRAY",
    "IIMAGE1D",
    "IIMAGE2D",
    "IIMAGE3D",
    "IIMAGECUBE",
    "IIMAGE1DARRAY",
    "IIMAGE2DARRAY",
    "IIMAGECUBEARRAY",
    "IIMAGE2DRECT",
    "IIMAGEBUFFER",
    "IIMAGE2DMS",
    "IIMAGE2DMSARRAY",
    "UIMAGE1D",
    "UIMAGE2D",
    "UIMAGE3D",
    "UIMAGECUBE",
    "UIMAGE1DARRAY",
    "UIMAGE2DARRAY",
    "UIMAGECUBEARRAY",
    "UIMAGE2DRECT",
    "UIMAGEBUFFER",
    "UIMAGE2DMS",
    "UIMAGE2DMSARRAY",
    "ATOMIC_UINT",
}

QUALIFIER_TOKENS = {
    "IN",
    "OUT",
    "INOUT",
    "UNIFORM",
    "CONST",
    "ATTRIBUTE",
    "VARYING",
    "BUFFER",
    "SHARED",
    "READONLY",
    "WRITEONLY",
    "COHERENT",
    "VOLATILE",
    "RESTRICT",
    "FLAT",
    "SMOOTH",
    "NOPERSPECTIVE",
    "CENTROID",
    "SAMPLE",
    "PATCH",
    "INVARIANT",
    "PRECISE",
    "SUBROUTINE",
    "LOWP",
    "MEDIUMP",
    "HIGHP",
}

ASSIGNMENT_TOKENS = {
    "EQUALS": "=",
    "PLUS_EQUALS": "+=",
    "MINUS_EQUALS": "-=",
    "MULTIPLY_EQUALS": "*=",
    "DIVIDE_EQUALS": "/=",
    "MOD_EQUALS": "%=",
    "ASSIGN_AND": "&=",
    "ASSIGN_OR": "|=",
    "ASSIGN_XOR": "^=",
    "ASSIGN_SHIFT_LEFT": "<<=",
    "ASSIGN_SHIFT_RIGHT": ">>=",
}


class GLSLParser:
    def __init__(self, tokens, shader_type="vertex"):
        self.tokens = tokens or [("EOF", "")]
        self.shader_type = shader_type
        self.index = 0
        self.current_token = self.tokens[self.index]

    def advance(self):
        self.index += 1
        if self.index < len(self.tokens):
            self.current_token = self.tokens[self.index]
        else:
            self.current_token = ("EOF", "")

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.advance()
        else:
            raise SyntaxError(
                f"Expected {token_type}, got {self.current_token[0]} ({self.current_token[1]})"
            )

    def peek(self, offset=1):
        idx = self.index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return ("EOF", "")

    def skip_newlines(self):
        while self.current_token[0] == "NEWLINE":
            self.advance()

    def parse(self):
        shader = self.parse_shader()
        if self.current_token[0] != "EOF":
            self.eat("EOF")
        return shader

    def parse_shader(self):
        io_variables = []
        uniforms = []
        constants = []
        global_variables = []
        functions = []
        structs = []
        preprocessor = []
        layouts = []

        while self.current_token[0] != "EOF":
            self.skip_newlines()
            if self.current_token[0] == "EOF":
                break

            if self.current_token[0] == "HASH":
                preprocessor.append(self.parse_preprocessor())
                continue

            if self.current_token[0] == "PRECISION":
                precision_stmt = self.parse_precision_statement()
                if precision_stmt:
                    preprocessor.append(precision_stmt)
                continue

            if self.current_token[0] == "STRUCT":
                struct_node, extra_vars = self.parse_struct()
                structs.append(struct_node)
                for var in extra_vars:
                    global_variables.append(var)
                continue

            layout = None
            if self.current_token[0] == "LAYOUT":
                layout = self.parse_layout_qualifier()

            qualifiers = self.parse_qualifiers()

            if layout is not None and self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                layouts.append({"layout": layout, "qualifiers": qualifiers})
                continue

            if (
                qualifiers
                and self.current_token[0] == "IDENTIFIER"
                and self.peek(1)[0] == "SEMICOLON"
            ):
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                self.eat("SEMICOLON")
                global_variables.append(
                    VariableNode("", name, qualifiers=qualifiers, layout=layout)
                )
                continue

            if self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "LBRACE":
                struct_node, block_vars = self.parse_interface_block(qualifiers, layout)
                structs.append(struct_node)
                for var in block_vars:
                    lowered = {q.lower() for q in var.qualifiers or []}
                    if "uniform" in lowered:
                        uniforms.append(var)
                    elif "in" in lowered or "out" in lowered or "inout" in lowered:
                        io_variables.append(var)
                    else:
                        global_variables.append(var)
                continue

            if self.current_token[0] == "STRUCT":
                struct_node, extra_vars = self.parse_struct()
                structs.append(struct_node)
                for var in extra_vars:
                    global_variables.append(var)
                continue

            if (
                self.current_token[0] in TYPE_TOKENS
                or self.current_token[0] == "IDENTIFIER"
            ):
                type_name = self.parse_type()

                if (
                    self.current_token[0] == "IDENTIFIER"
                    and self.peek(1)[0] == "LPAREN"
                ):
                    function = self.parse_function(type_name)
                    functions.append(function)
                    continue

                declarations = self.parse_variable_declarations(
                    type_name, qualifiers=qualifiers, layout=layout
                )

                for var in declarations:
                    lowered = {q.lower() for q in var.qualifiers or []}
                    if "uniform" in lowered:
                        uniforms.append(var)
                    elif "const" in lowered:
                        constants.append(var)
                    elif "in" in lowered or "out" in lowered or "inout" in lowered:
                        io_variables.append(var)
                    else:
                        global_variables.append(var)
                continue

            # Skip unexpected tokens to avoid infinite loop
            self.advance()

        shader = ShaderNode(
            functions=functions,
            structs=structs,
            global_variables=global_variables,
            uniforms=uniforms,
            io_variables=io_variables,
            constant=constants,
            shader_type=self.shader_type,
            preprocessor=preprocessor,
            layouts=layouts,
        )
        return shader

    def parse_preprocessor(self):
        self.eat("HASH")
        tokens = ["#"]
        while self.current_token[0] not in ("NEWLINE", "EOF"):
            tokens.append(self.current_token[1])
            self.advance()
        if self.current_token[0] == "NEWLINE":
            self.advance()
        if len(tokens) > 1:
            return "#" + " ".join(tokens[1:]).strip()
        return "#"

    def parse_precision_statement(self):
        parts = [self.current_token[1]]
        self.eat("PRECISION")
        while self.current_token[0] != "SEMICOLON" and self.current_token[0] != "EOF":
            parts.append(self.current_token[1])
            self.advance()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return " ".join(parts).strip() + ";"

    def parse_layout_qualifier(self):
        qualifiers = {}
        self.eat("LAYOUT")
        self.eat("LPAREN")
        while self.current_token[0] != "RPAREN":
            if self.current_token[0] in ("IDENTIFIER", "IN", "OUT") or (
                self.current_token[0] in QUALIFIER_TOKENS
            ):
                key = self.current_token[1]
                self.advance()
                value = None
                if self.current_token[0] == "EQUALS":
                    self.eat("EQUALS")
                    value = self.parse_layout_value()
                qualifiers[key] = value
            else:
                raise SyntaxError(
                    f"Unexpected token in layout qualifier: {self.current_token}"
                )
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        return qualifiers

    def parse_layout_value(self):
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        if self.current_token[0] == "IDENTIFIER":
            value = self.current_token[1]
            self.eat("IDENTIFIER")
            return value
        raise SyntaxError(f"Expected layout qualifier value, got {self.current_token}")

    def parse_qualifiers(self):
        qualifiers = []
        while self.current_token[0] in QUALIFIER_TOKENS:
            if self.current_token[0] == "SUBROUTINE":
                self.advance()
                if self.current_token[0] == "LPAREN":
                    self.eat("LPAREN")
                    if self.current_token[0] in TYPE_TOKENS:
                        type_name = self.current_token[1]
                        self.advance()
                    elif self.current_token[0] == "IDENTIFIER":
                        type_name = self.current_token[1]
                        self.advance()
                    else:
                        raise SyntaxError(
                            f"Expected subroutine type, got {self.current_token}"
                        )
                    self.eat("RPAREN")
                    qualifiers.append(f"subroutine({type_name})")
                else:
                    qualifiers.append("subroutine")
                continue
            qualifiers.append(self.current_token[1])
            self.advance()
        return qualifiers

    def parse_type(self):
        if self.current_token[0] in TYPE_TOKENS:
            type_name = self.current_token[1]
            self.advance()
            return type_name
        if self.current_token[0] == "IDENTIFIER":
            type_name = self.current_token[1]
            self.eat("IDENTIFIER")
            return type_name
        raise SyntaxError(f"Expected type, got {self.current_token}")

    def parse_variable_declarations(
        self, type_name, qualifiers=None, layout=None, consume_semicolon=True
    ):
        variables = []
        while True:
            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected identifier in declaration, got {self.current_token}"
                )

            name = self.current_token[1]
            self.eat("IDENTIFIER")

            array_size = None
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] != "RBRACKET":
                    array_size = self.parse_expression()
                self.eat("RBRACKET")

            value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                value = self.parse_expression()

            var = VariableNode(
                type_name,
                name,
                value=value,
                qualifiers=qualifiers or [],
                array_size=array_size,
                layout=layout,
            )

            lowered = {q.lower() for q in qualifiers or []}
            if "in" in lowered:
                var.io_type = "IN"
            if "out" in lowered:
                var.io_type = "OUT"
            if "inout" in lowered:
                var.io_type = "INOUT"
            if "const" in lowered:
                var.is_const = True

            variables.append(var)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

        if consume_semicolon:
            if self.current_token[0] != "SEMICOLON":
                raise SyntaxError(
                    f"Expected ';' after declaration, got {self.current_token}"
                )
            self.eat("SEMICOLON")
        return variables

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            self.skip_newlines()
            if self.current_token[0] == "RBRACE":
                break
            if self.current_token[0] in QUALIFIER_TOKENS:
                qualifiers = self.parse_qualifiers()
            else:
                qualifiers = []
            member_type = self.parse_type()

            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected identifier in struct field, got {self.current_token}"
                )

            member_name = self.current_token[1]
            self.eat("IDENTIFIER")

            array_size = None
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] != "RBRACKET":
                    array_size = self.parse_expression()
                self.eat("RBRACKET")

            self.eat("SEMICOLON")
            members.append(
                VariableNode(
                    member_type,
                    member_name,
                    qualifiers=qualifiers,
                    array_size=array_size,
                )
            )

        self.eat("RBRACE")

        variables = []
        if self.current_token[0] == "IDENTIFIER":
            variables = self.parse_variable_declarations(name, qualifiers=[])
        else:
            self.eat("SEMICOLON")

        return StructNode(name, members), variables

    def parse_interface_block(self, qualifiers, layout):
        block_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            self.skip_newlines()
            if self.current_token[0] == "RBRACE":
                break
            member_qualifiers = self.parse_qualifiers()
            member_type = self.parse_type()

            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected identifier in interface block, got {self.current_token}"
                )
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")

            array_size = None
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] != "RBRACKET":
                    array_size = self.parse_expression()
                self.eat("RBRACKET")

            self.eat("SEMICOLON")
            members.append(
                VariableNode(
                    member_type,
                    member_name,
                    qualifiers=member_qualifiers,
                    array_size=array_size,
                )
            )

        self.eat("RBRACE")

        instance_name = None
        array_size = None
        if self.current_token[0] == "IDENTIFIER":
            instance_name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] != "RBRACKET":
                    array_size = self.parse_expression()
                self.eat("RBRACKET")

        self.eat("SEMICOLON")

        struct_node = StructNode(block_name, members)
        block_vars = []

        if instance_name:
            block_vars.append(
                VariableNode(
                    block_name,
                    instance_name,
                    qualifiers=qualifiers,
                    array_size=array_size,
                    layout=layout,
                )
            )
        else:
            for member in members:
                member.qualifiers = list(member.qualifiers or []) + list(
                    qualifiers or []
                )
                member.layout = layout
                block_vars.append(member)

        return struct_node, block_vars

    def parse_function(self, return_type):
        name = self.current_token[1]
        qualifier = None
        if name == "main":
            qualifier = self.shader_type
        self.eat("IDENTIFIER")
        params = self.parse_parameters()

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return FunctionNode(return_type, name, params, body=[])

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        qualifiers = [qualifier] if qualifier else []
        return FunctionNode(return_type, name, params, body, qualifiers=qualifiers)

    def parse_parameters(self):
        self.eat("LPAREN")
        params = []
        if self.current_token[0] != "RPAREN":
            while True:
                qualifiers = self.parse_qualifiers()
                param_type = self.parse_type()
                param_name = self.current_token[1]
                self.eat("IDENTIFIER")

                array_size = None
                if self.current_token[0] == "LBRACKET":
                    self.eat("LBRACKET")
                    if self.current_token[0] != "RBRACKET":
                        array_size = self.parse_expression()
                    self.eat("RBRACKET")

                params.append(
                    VariableNode(
                        param_type,
                        param_name,
                        qualifiers=qualifiers,
                        array_size=array_size,
                    )
                )
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break
        self.eat("RPAREN")
        return params

    def parse_block(self):
        statements = []
        while self.current_token[0] not in ("RBRACE", "EOF"):
            self.skip_newlines()
            if self.current_token[0] in ("RBRACE", "EOF"):
                break
            stmt = self.parse_statement()
            if isinstance(stmt, list):
                statements.extend(stmt)
            elif stmt is not None:
                statements.append(stmt)
        return statements

    def parse_statement(self):
        self.skip_newlines()
        if self.current_token[0] in ("RBRACE", "EOF"):
            return None
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            block = self.parse_block()
            self.eat("RBRACE")
            return BlockNode(block)
        if self.current_token[0] == "IF":
            return self.parse_if_statement()
        if self.current_token[0] == "FOR":
            return self.parse_for_loop()
        if self.current_token[0] == "WHILE":
            return self.parse_while_loop()
        if self.current_token[0] == "DO":
            return self.parse_do_while_loop()
        if self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()
        if self.current_token[0] == "RETURN":
            return self.parse_return_statement()
        if self.current_token[0] == "BREAK":
            self.eat("BREAK")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return BreakNode()
        if self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return ContinueNode()
        if self.current_token[0] == "DISCARD":
            self.eat("DISCARD")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return DiscardNode()

        # Variable declaration
        if (
            self.current_token[0] in QUALIFIER_TOKENS
            or self.current_token[0] in TYPE_TOKENS
        ):
            qualifiers = self.parse_qualifiers()
            type_name = self.parse_type()
            return self.parse_variable_declarations(type_name, qualifiers=qualifiers)
        if self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "IDENTIFIER":
            type_name = self.parse_type()
            return self.parse_variable_declarations(type_name, qualifiers=[])

        # Expression / assignment
        expr = self.parse_expression()
        if self.current_token[0] in ASSIGNMENT_TOKENS:
            op = ASSIGNMENT_TOKENS[self.current_token[0]]
            self.eat(self.current_token[0])
            right = self.parse_expression()
            if self.current_token[0] != "SEMICOLON":
                raise SyntaxError(
                    f"Expected ';' after assignment, got {self.current_token}"
                )
            self.eat("SEMICOLON")
            return AssignmentNode(expr, right, op)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return expr
        raise SyntaxError(f"Expected ';' after expression, got {self.current_token}")

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        if_body = self.parse_block()
        self.eat("RBRACE")

        else_body = None
        else_if_chain = []
        while self.current_token[0] == "ELSE" and self.peek(1)[0] == "IF":
            self.eat("ELSE")
            self.eat("IF")
            self.eat("LPAREN")
            else_if_condition = self.parse_expression()
            self.eat("RPAREN")
            self.eat("LBRACE")
            else_if_body = self.parse_block()
            self.eat("RBRACE")
            else_if_chain.append((else_if_condition, else_if_body))

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            self.eat("LBRACE")
            else_body = self.parse_block()
            self.eat("RBRACE")

        node = IfNode(condition, if_body, else_body)
        if else_if_chain:
            node.else_if_chain = else_if_chain
        return node

    def parse_for_loop(self):
        self.eat("FOR")
        self.eat("LPAREN")

        init = None
        if self.current_token[0] != "SEMICOLON":
            if (
                self.current_token[0] in QUALIFIER_TOKENS
                or self.current_token[0] in TYPE_TOKENS
            ):
                qualifiers = self.parse_qualifiers()
                type_name = self.parse_type()
                init_decls = self.parse_variable_declarations(
                    type_name, qualifiers=qualifiers, consume_semicolon=False
                )
                init = init_decls[0] if init_decls else None
            else:
                init = self.parse_expression()
                if self.current_token[0] in ASSIGNMENT_TOKENS:
                    op = ASSIGNMENT_TOKENS[self.current_token[0]]
                    self.eat(self.current_token[0])
                    right = self.parse_expression()
                    init = AssignmentNode(init, right, op)

        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression()
            if self.current_token[0] in ASSIGNMENT_TOKENS:
                op = ASSIGNMENT_TOKENS[self.current_token[0]]
                self.eat(self.current_token[0])
                right = self.parse_expression()
                update = AssignmentNode(update, right, op)
        self.eat("RPAREN")

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return ForNode(init, condition, update, body)

    def parse_while_loop(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")
        return WhileNode(condition, body)

    def parse_do_while_loop(self):
        self.eat("DO")
        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        default_statements = None

        while self.current_token[0] not in ("RBRACE", "EOF"):
            self.skip_newlines()
            if self.current_token[0] in ("RBRACE", "EOF"):
                break
            if self.current_token[0] == "CASE":
                cases.append(self.parse_case_statement())
            elif self.current_token[0] == "DEFAULT":
                self.eat("DEFAULT")
                self.eat("COLON")
                default_statements = []
                while self.current_token[0] not in (
                    "CASE",
                    "DEFAULT",
                    "RBRACE",
                    "EOF",
                ):
                    self.skip_newlines()
                    if self.current_token[0] in ("CASE", "DEFAULT", "RBRACE", "EOF"):
                        break
                    stmt = self.parse_statement()
                    if isinstance(stmt, list):
                        default_statements.extend(stmt)
                    elif stmt is not None:
                        default_statements.append(stmt)
            else:
                raise SyntaxError(
                    f"Unexpected token in switch statement: {self.current_token}"
                )

        self.eat("RBRACE")
        return SwitchNode(expression, cases, default_statements)

    def parse_case_statement(self):
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        statements = []
        while self.current_token[0] not in ("CASE", "DEFAULT", "RBRACE", "EOF"):
            self.skip_newlines()
            if self.current_token[0] in ("CASE", "DEFAULT", "RBRACE", "EOF"):
                break
            stmt = self.parse_statement()
            if isinstance(stmt, list):
                statements.extend(stmt)
            elif stmt is not None:
                statements.append(stmt)
        return CaseNode(value, statements)

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode()
        value = self.parse_expression()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_expression(self):
        self.skip_newlines()
        return self.parse_ternary()

    def parse_ternary(self):
        expr = self.parse_logical_or()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(expr, true_expr, false_expr)
        return expr

    def parse_logical_or(self):
        expr = self.parse_logical_and()
        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_logical_and(self):
        expr = self.parse_bitwise_or()
        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_bitwise_or(self):
        expr = self.parse_bitwise_xor()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_bitwise_xor(self):
        expr = self.parse_bitwise_and()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_bitwise_and(self):
        expr = self.parse_equality()
        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_equality(self):
        expr = self.parse_relational()
        while self.current_token[0] in ("EQUAL", "NOT_EQUAL"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_relational(self):
        expr = self.parse_shift()
        while self.current_token[0] in (
            "LESS_THAN",
            "LESS_EQUAL",
            "GREATER_THAN",
            "GREATER_EQUAL",
        ):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_shift(self):
        expr = self.parse_additive()
        while self.current_token[0] in ("SHIFT_LEFT", "SHIFT_RIGHT"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_additive(self):
        expr = self.parse_multiplicative()
        while self.current_token[0] in ("PLUS", "MINUS"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_multiplicative(self):
        expr = self.parse_unary()
        while self.current_token[0] in ("MULTIPLY", "DIVIDE", "MOD"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_unary(self):
        if self.current_token[0] in ("PLUS", "MINUS", "LOGICAL_NOT", "BITWISE_NOT"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        if self.current_token[0] in ("INCREMENT", "DECREMENT"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            if (
                self.current_token[0] == "LBRACKET"
                and self.peek(1)[0] == "RBRACKET"
                and self.peek(2)[0] == "LPAREN"
                and isinstance(expr, VariableNode)
            ):
                self.eat("LBRACKET")
                self.eat("RBRACKET")
                expr = VariableNode("", f"{expr.name}[]")
                continue
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                expr = ArrayAccessNode(expr, index)
                continue
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                expr = MemberAccessNode(expr, member)
                continue
            if self.current_token[0] == "LPAREN":
                args = self.parse_call_arguments()
                expr = FunctionCallNode(expr, args)
                continue
            if self.current_token[0] in ("INCREMENT", "DECREMENT"):
                op = self.current_token[1]
                self.eat(self.current_token[0])
                expr = PostfixOpNode(expr, op)
                continue
            break
        return expr

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

    def parse_primary(self):
        self.skip_newlines()
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return NumberNode(value)
        if self.current_token[0] in ("TRUE", "FALSE"):
            value = self.current_token[1]
            self.advance()
            return value
        if (
            self.current_token[0] in TYPE_TOKENS
            or self.current_token[0] == "IDENTIFIER"
        ):
            name = self.current_token[1]
            self.advance()
            return VariableNode("", name)
        if self.current_token[0] in ("STRING", "CHAR_LITERAL"):
            value = self.current_token[1]
            self.advance()
            return value
        raise SyntaxError(f"Unexpected token in expression: {self.current_token}")

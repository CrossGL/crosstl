"""Parser for GLSL source AST construction."""

from .OpenglAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    NumberNode,
    PostfixOpNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
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

RAY_STORAGE_QUALIFIERS = {
    "rayPayloadEXT",
    "rayPayloadInEXT",
    "hitAttributeEXT",
    "callableDataEXT",
    "callableDataInEXT",
}

MESH_STORAGE_QUALIFIERS = {
    "perprimitiveEXT",
    "pervertexEXT",
    "perviewEXT",
    "taskPayloadSharedEXT",
}

IDENTIFIER_QUALIFIERS = RAY_STORAGE_QUALIFIERS | MESH_STORAGE_QUALIFIERS

NAME_TOKENS = {"IDENTIFIER", "SAMPLE"}

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
    """Parse GLSL tokens into the OpenGL backend shader AST."""

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

    def peek_non_newline(self, offset=1):
        idx = self.index + offset
        while idx < len(self.tokens) and self.tokens[idx][0] == "NEWLINE":
            idx += 1
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
                self.skip_newlines()

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

            if (
                self.current_token[0] == "IDENTIFIER"
                and self.peek_non_newline()[0] == "LBRACE"
            ):
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
                struct_node, extra_vars = self.parse_struct(
                    qualifiers=qualifiers, layout=layout
                )
                structs.append(struct_node)
                for var in extra_vars:
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

            if (
                self.current_token[0] in TYPE_TOKENS
                or self.current_token[0] == "IDENTIFIER"
            ):
                type_name = self.parse_type()
                self.skip_newlines()
                type_array_sizes = []
                if self.current_token[0] == "LBRACKET":
                    type_array_sizes = self.parse_array_suffixes()
                    self.skip_newlines()

                if (
                    self.current_token[0] == "IDENTIFIER"
                    and self.peek(1)[0] == "LPAREN"
                ):
                    function = self.parse_function(
                        self.type_name_with_array_suffixes(type_name, type_array_sizes)
                    )
                    functions.append(function)
                    continue

                declarations = self.parse_variable_declarations(
                    type_name,
                    qualifiers=qualifiers,
                    layout=layout,
                    type_array_sizes=type_array_sizes,
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
        value = self.parse_layout_constant_expression()
        if isinstance(value, NumberNode):
            return value.value
        if isinstance(value, VariableNode) and not value.vtype:
            return value.name
        return value

    def parse_layout_constant_expression(self):
        expr = self.parse_logical_or()
        self.skip_newlines()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_layout_constant_expression()
            self.skip_newlines()
            self.eat("COLON")
            false_expr = self.parse_layout_constant_expression()
            return TernaryOpNode(expr, true_expr, false_expr)
        return expr

    def parse_qualifiers(self):
        qualifiers = []
        while self.current_token[0] in QUALIFIER_TOKENS or (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in IDENTIFIER_QUALIFIERS
        ):
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
            return type_name + self.parse_type_template_suffix()
        if self.current_token[0] == "IDENTIFIER":
            type_name = self.current_token[1]
            self.eat("IDENTIFIER")
            return type_name + self.parse_type_template_suffix()
        raise SyntaxError(f"Expected type, got {self.current_token}")

    def parse_type_template_suffix(self):
        if self.current_token[0] != "LESS_THAN":
            return ""

        self.eat("LESS_THAN")
        parts = []
        depth = 1
        while depth:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated type template argument list")
            if self.current_token[0] == "LESS_THAN":
                depth += 1
                parts.append("<")
                self.advance()
                continue
            if self.current_token[0] == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.advance()
                    break
                parts.append(">")
                self.advance()
                continue
            parts.append(self.current_token[1])
            self.advance()

        return f"<{self.format_type_template_parts(parts)}>"

    def format_type_template_parts(self, parts):
        text = " ".join(str(part) for part in parts)
        return (
            text.replace(" ,", ",")
            .replace(", ", ", ")
            .replace("< ", "<")
            .replace(" >", ">")
        )

    def format_type_array_suffixes(self, array_sizes):
        return "".join(
            f"[{self.format_type_array_size(size)}]" if size is not None else "[]"
            for size in array_sizes
        )

    def format_type_array_size(self, size):
        if isinstance(size, NumberNode):
            return size.value
        if isinstance(size, VariableNode) and not size.vtype:
            return size.name
        if isinstance(size, BinaryOpNode):
            left = self.format_type_array_size(size.left)
            right = self.format_type_array_size(size.right)
            return f"{left} {size.op} {right}"
        if isinstance(size, UnaryOpNode):
            return f"{size.op}{self.format_type_array_size(size.operand)}"
        if isinstance(size, TernaryOpNode):
            condition = self.format_type_array_size(size.condition)
            true_expr = self.format_type_array_size(size.true_expr)
            false_expr = self.format_type_array_size(size.false_expr)
            return f"{condition} ? {true_expr} : {false_expr}"
        if isinstance(size, FunctionCallNode):
            name = self.format_type_array_size(size.name)
            args = ", ".join(self.format_type_array_size(arg) for arg in size.args)
            return f"{name}({args})"
        return str(size)

    def type_name_with_array_suffixes(self, type_name, array_sizes):
        if not array_sizes:
            return type_name
        return f"{type_name}{self.format_type_array_suffixes(array_sizes)}"

    def is_name_token(self):
        return self.current_token[0] in NAME_TOKENS

    def parse_identifier_name(self, context="identifier"):
        if self.is_name_token():
            name = self.current_token[1]
            self.advance()
            return name
        raise SyntaxError(f"Expected {context}, got {self.current_token}")

    def is_declaration_start(self):
        if self.current_token[0] in TYPE_TOKENS:
            return True
        if self.current_token[0] not in QUALIFIER_TOKENS:
            return False
        if self.current_token[0] in NAME_TOKENS:
            next_token = self.peek_non_newline()
            return next_token[0] in QUALIFIER_TOKENS or next_token[0] in TYPE_TOKENS
        return True

    def parse_variable_declarations(
        self,
        type_name,
        qualifiers=None,
        layout=None,
        consume_semicolon=True,
        type_array_sizes=None,
    ):
        variables = []
        if type_array_sizes is None:
            type_array_sizes = []
            if self.current_token[0] == "LBRACKET":
                type_array_sizes = self.parse_array_suffixes()
        while True:
            self.skip_newlines()
            if not self.is_name_token():
                raise SyntaxError(
                    f"Expected identifier in declaration, got {self.current_token}"
                )

            name = self.parse_identifier_name()

            array_sizes = list(type_array_sizes) + self.parse_array_suffixes()
            is_array = bool(array_sizes)
            array_size = array_sizes[0] if array_sizes else None

            value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                value = self.parse_assignment_expression()

            var = VariableNode(
                type_name,
                name,
                value=value,
                qualifiers=qualifiers or [],
                array_size=array_size,
                layout=layout,
                is_array=is_array,
                array_sizes=array_sizes,
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

            self.skip_newlines()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_newlines()
                continue
            break

        if consume_semicolon:
            if self.current_token[0] != "SEMICOLON":
                raise SyntaxError(
                    f"Expected ';' after declaration, got {self.current_token}"
                )
            self.eat("SEMICOLON")
        return variables

    def parse_array_suffixes(self):
        sizes = []
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            sizes.append(size)
        return sizes

    def parse_struct(self, qualifiers=None, layout=None):
        declaration_qualifiers = qualifiers or []
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.skip_newlines()
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            self.skip_newlines()
            if self.current_token[0] == "RBRACE":
                break
            member_layout = None
            if self.current_token[0] == "LAYOUT":
                member_layout = self.parse_layout_qualifier()
                self.skip_newlines()
            member_qualifiers = self.parse_qualifiers()
            member_type = self.parse_type()
            members.extend(
                self.parse_variable_declarations(
                    member_type,
                    qualifiers=member_qualifiers,
                    layout=member_layout,
                )
            )

        self.eat("RBRACE")
        self.skip_newlines()

        variables = []
        if self.current_token[0] == "IDENTIFIER":
            variables = self.parse_variable_declarations(
                name, qualifiers=declaration_qualifiers, layout=layout
            )
        else:
            self.eat("SEMICOLON")

        return StructNode(name, members), variables

    def parse_interface_block(self, qualifiers, layout):
        block_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.skip_newlines()
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            self.skip_newlines()
            if self.current_token[0] == "RBRACE":
                break
            member_layout = None
            if self.current_token[0] == "LAYOUT":
                member_layout = self.parse_layout_qualifier()
                self.skip_newlines()
            member_qualifiers = self.parse_qualifiers()
            member_type = self.parse_type()
            member_nodes = self.parse_variable_declarations(
                member_type,
                qualifiers=member_qualifiers,
                layout=member_layout,
            )
            for member_node in member_nodes:
                member_node.interface_block = block_name
                members.append(member_node)

        self.eat("RBRACE")
        self.skip_newlines()

        instance_name = None
        array_size = None
        instance_is_array = False
        if self.current_token[0] == "IDENTIFIER":
            instance_name = self.current_token[1]
            self.eat("IDENTIFIER")
            array_sizes = self.parse_array_suffixes()
            instance_is_array = bool(array_sizes)
            array_size = array_sizes[0] if array_sizes else None
        else:
            array_sizes = []

        self.skip_newlines()
        self.eat("SEMICOLON")

        struct_node = StructNode(block_name, members)
        struct_node.interface_block = True
        struct_node.interface_qualifiers = list(qualifiers or [])
        struct_node.interface_layout = layout
        struct_node.interface_instance_name = instance_name
        struct_node.interface_instance_is_array = instance_is_array
        struct_node.interface_array_size = array_size
        struct_node.interface_array_sizes = array_sizes
        block_vars = []

        if instance_name:
            block_var = VariableNode(
                block_name,
                instance_name,
                qualifiers=qualifiers,
                array_size=array_size,
                layout=layout,
                array_sizes=array_sizes,
            )
            block_var.interface_block = block_name
            block_var.is_array = instance_is_array
            block_vars.append(block_var)
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
        self.skip_newlines()

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
        self.skip_newlines()
        if self.current_token[0] == "VOID":
            lookahead = 1
            while self.peek(lookahead)[0] == "NEWLINE":
                lookahead += 1
            if self.peek(lookahead)[0] == "RPAREN":
                self.eat("VOID")
                self.skip_newlines()
                self.eat("RPAREN")
                return params
        if self.current_token[0] != "RPAREN":
            while True:
                self.skip_newlines()
                qualifiers = self.parse_qualifiers()
                param_type = self.parse_type()
                type_array_sizes = []
                if self.current_token[0] == "LBRACKET":
                    type_array_sizes = self.parse_array_suffixes()

                if self.current_token[0] in ("COMMA", "RPAREN"):
                    param_name = f"_param{len(params)}"
                    array_sizes = type_array_sizes
                else:
                    param_name = self.parse_identifier_name("parameter name")
                    array_sizes = type_array_sizes + self.parse_array_suffixes()

                array_size = array_sizes[0] if array_sizes else None

                params.append(
                    VariableNode(
                        param_type,
                        param_name,
                        qualifiers=qualifiers,
                        array_size=array_size,
                        array_sizes=array_sizes,
                        is_array=bool(array_sizes),
                    )
                )
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break
        self.skip_newlines()
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

    def parse_statement_or_block(self):
        self.skip_newlines()
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            body = self.parse_block()
            self.eat("RBRACE")
            return body
        stmt = self.parse_statement()
        if stmt is None:
            return []
        if isinstance(stmt, list):
            return stmt
        return [stmt]

    def parse_statement(self):
        self.skip_newlines()
        if self.current_token[0] in ("RBRACE", "EOF"):
            return None
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
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
        if self.current_token[0] == "STRUCT":
            struct_node, extra_vars = self.parse_struct()
            return [struct_node, *extra_vars]

        if self.is_declaration_start():
            qualifiers = self.parse_qualifiers()
            type_name = self.parse_type()
            self.skip_newlines()
            return self.parse_variable_declarations(type_name, qualifiers=qualifiers)
        if self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "IDENTIFIER":
            type_name = self.parse_type()
            self.skip_newlines()
            return self.parse_variable_declarations(type_name, qualifiers=[])

        expr = self.parse_expression()
        self.skip_newlines()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return expr
        raise SyntaxError(f"Expected ';' after expression, got {self.current_token}")

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_statement_or_block()
        self.skip_newlines()

        else_body = None
        else_if_chain = []
        while self.current_token[0] == "ELSE" and self.peek_non_newline()[0] == "IF":
            self.eat("ELSE")
            self.skip_newlines()
            self.eat("IF")
            self.eat("LPAREN")
            else_if_condition = self.parse_expression()
            self.eat("RPAREN")
            else_if_body = self.parse_statement_or_block()
            else_if_chain.append((else_if_condition, else_if_body))
            self.skip_newlines()

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement_or_block()

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
                init = self.parse_assignment_expression()

        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            updates = []
            while True:
                updates.append(self.parse_assignment_expression())
                self.skip_newlines()
                if self.current_token[0] != "COMMA":
                    break
                self.eat("COMMA")
                self.skip_newlines()
            update = updates[0] if len(updates) == 1 else updates
        self.eat("RPAREN")

        body = self.parse_statement_or_block()

        return ForNode(init, condition, update, body)

    def parse_while_loop(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement_or_block()
        return WhileNode(condition, body)

    def parse_do_while_loop(self):
        self.eat("DO")
        self.skip_newlines()
        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")
        self.skip_newlines()
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
        self.skip_newlines()
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
        return self.parse_comma_expression()

    def parse_comma_expression(self):
        expr = self.parse_assignment_expression()
        self.skip_newlines()
        while self.current_token[0] == "COMMA":
            op = self.current_token[1]
            self.eat("COMMA")
            right = self.parse_assignment_expression()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_assignment_expression(self):
        expr = self.parse_ternary()
        self.skip_newlines()
        if self.current_token[0] in ASSIGNMENT_TOKENS:
            op = ASSIGNMENT_TOKENS[self.current_token[0]]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(expr, right, op)
        return expr

    def parse_ternary(self):
        expr = self.parse_logical_or()
        self.skip_newlines()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.skip_newlines()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(expr, true_expr, false_expr)
        return expr

    def parse_logical_or(self):
        expr = self.parse_logical_xor()
        self.skip_newlines()
        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_xor()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_logical_xor(self):
        expr = self.parse_logical_and()
        self.skip_newlines()
        while self.current_token[0] == "LOGICAL_XOR":
            op = self.current_token[1]
            self.eat("LOGICAL_XOR")
            right = self.parse_logical_and()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_logical_and(self):
        expr = self.parse_bitwise_or()
        self.skip_newlines()
        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_bitwise_or(self):
        expr = self.parse_bitwise_xor()
        self.skip_newlines()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_bitwise_xor(self):
        expr = self.parse_bitwise_and()
        self.skip_newlines()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_bitwise_and(self):
        expr = self.parse_equality()
        self.skip_newlines()
        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_equality(self):
        expr = self.parse_relational()
        self.skip_newlines()
        while self.current_token[0] in ("EQUAL", "NOT_EQUAL"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_relational(self):
        expr = self.parse_shift()
        self.skip_newlines()
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
            self.skip_newlines()
        return expr

    def parse_shift(self):
        expr = self.parse_additive()
        self.skip_newlines()
        while self.current_token[0] in ("SHIFT_LEFT", "SHIFT_RIGHT"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_additive(self):
        expr = self.parse_multiplicative()
        self.skip_newlines()
        while self.current_token[0] in ("PLUS", "MINUS"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
        return expr

    def parse_multiplicative(self):
        expr = self.parse_unary()
        self.skip_newlines()
        while self.current_token[0] in ("MULTIPLY", "DIVIDE", "MOD"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()
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
            if self.is_array_constructor_suffix(expr):
                while self.current_token[0] == "LBRACKET":
                    self.eat("LBRACKET")
                    if self.current_token[0] != "RBRACKET":
                        self.parse_expression()
                    self.eat("RBRACKET")
                    self.skip_newlines()
                args = self.parse_call_arguments()
                expr = InitializerListNode(args)
                continue
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                expr = ArrayAccessNode(expr, index)
                continue
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.parse_identifier_name("member name")
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

    def is_array_constructor_suffix(self, expr):
        if not isinstance(expr, VariableNode) or self.current_token[0] != "LBRACKET":
            return False

        idx = self.index
        saw_suffix = False
        while idx < len(self.tokens):
            while idx < len(self.tokens) and self.tokens[idx][0] == "NEWLINE":
                idx += 1
            if idx >= len(self.tokens) or self.tokens[idx][0] != "LBRACKET":
                break

            saw_suffix = True
            depth = 0
            while idx < len(self.tokens):
                token_type = self.tokens[idx][0]
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
                idx += 1
            else:
                return False

        while idx < len(self.tokens) and self.tokens[idx][0] == "NEWLINE":
            idx += 1
        return saw_suffix and idx < len(self.tokens) and self.tokens[idx][0] == "LPAREN"

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []
        self.skip_newlines()
        if self.current_token[0] != "RPAREN":
            while True:
                args.append(self.parse_assignment_expression())
                self.skip_newlines()
                if self.current_token[0] != "COMMA":
                    break
                self.eat("COMMA")
                self.skip_newlines()
        self.eat("RPAREN")
        return args

    def parse_primary(self):
        self.skip_newlines()
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        if self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return NumberNode(value)
        if self.current_token[0] in ("TRUE", "FALSE"):
            value = self.current_token[1]
            self.advance()
            return value
        if self.current_token[0] in TYPE_TOKENS or self.current_token[0] in NAME_TOKENS:
            name = self.current_token[1]
            self.advance()
            return VariableNode("", name)
        if self.current_token[0] in ("STRING", "CHAR_LITERAL"):
            value = self.current_token[1]
            self.advance()
            return value
        raise SyntaxError(f"Unexpected token in expression: {self.current_token}")

    def parse_initializer_list(self):
        self.eat("LBRACE")
        elements = []
        self.skip_newlines()
        while self.current_token[0] != "RBRACE":
            elements.append(self.parse_assignment_expression())
            self.skip_newlines()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                self.skip_newlines()
                if self.current_token[0] == "RBRACE":
                    break
                continue
            break
        self.eat("RBRACE")
        return InitializerListNode(elements)

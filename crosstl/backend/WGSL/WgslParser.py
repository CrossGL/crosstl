"""The WGSL parser: a recursive-descent parser for the grammar of W3C WGSL, section 16, building :mod:`.WgslAst`.

Binary operators are parsed with C precedence. WGSL is stricter (``a & b == c`` and ``a | b ^ c`` need parentheses,
section 8.19), so every module WGSL accepts parses to the tree WGSL gives it; this parser does not refuse the mixed
forms WGSL rejects, which the reference validators do.
"""

from __future__ import annotations

from . import WgslAst as A
from .WgslLexer import Token, WgslLexer, WgslSyntaxError

__all__ = ["WgslParser"]

# Binary operators by precedence, loosest first.
_BINARY_LEVELS = (
    ("||",),
    ("&&",),
    ("|",),
    ("^",),
    ("&",),
    ("==", "!="),
    ("<", ">", "<=", ">="),
    ("<<", ">>"),
    ("+", "-"),
    ("*", "/", "%"),
)
_COMPOUND_ASSIGNMENTS = ("+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ">>=", "<<=")
_STATEMENT_KEYWORDS = ("if", "switch", "loop", "for", "while")


class WgslParser:
    """Parses the tokens of :class:`~.WgslLexer.WgslLexer` into a :class:`~.WgslAst.Module`."""

    def __init__(self, tokens, file_path=None):
        if isinstance(tokens, str):
            tokens = WgslLexer(tokens).tokenize()
        self.tokens: list[Token] = list(tokens)
        self.pos = 0
        self.file_path = file_path

    # Token helpers.

    @property
    def token(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        token = self.token
        if token.kind != "eof":
            self.pos += 1
        return token

    def error(self, message: str):
        return WgslSyntaxError(message, self.token.line)

    def expect_op(self, text: str) -> Token:
        if not self.token.is_op(text):
            raise self.error(
                f"expected '{text}', found '{self.token.text or 'end of file'}'"
            )
        return self.advance()

    def expect_keyword(self, text: str) -> Token:
        if not self.token.is_keyword(text):
            raise self.error(
                f"expected '{text}', found '{self.token.text or 'end of file'}'"
            )
        return self.advance()

    def expect_ident(self) -> str:
        if self.token.kind != "ident":
            raise self.error(
                f"expected an identifier, found '{self.token.text or 'end of file'}'"
            )
        return self.advance().text

    def accept_op(self, text: str) -> bool:
        if self.token.is_op(text):
            self.advance()
            return True
        return False

    # Module.

    def parse(self) -> A.Module:
        directives, declarations = [], []
        while self.token.kind == "keyword" and self.token.text in (
            "enable",
            "requires",
            "diagnostic",
        ):
            directives.append(self.parse_directive())
        while self.token.kind != "eof":
            if self.accept_op(";"):
                continue
            declarations.append(self.parse_global_declaration())
        return A.Module(directives, declarations)

    def parse_directive(self) -> A.Directive:
        line = self.token.line
        kind = self.advance().text
        if kind == "diagnostic":
            names = self.parse_diagnostic_control()
        else:
            names = [self.expect_ident()]
            while self.accept_op(",") and self.token.kind == "ident":
                names.append(self.expect_ident())
        self.expect_op(";")
        return A.Directive(kind, names, line)

    def parse_diagnostic_control(self) -> list:
        self.expect_op("(")
        names = [self.expect_ident()]
        self.expect_op(",")
        rule = self.expect_ident()
        if self.accept_op("."):
            rule += "." + self.expect_ident()
        names.append(rule)
        self.accept_op(",")
        self.expect_op(")")
        return names

    def parse_attributes(self) -> list:
        attributes = []
        while self.token.is_op("@"):
            line = self.advance().line
            name = self.advance().text
            args = []
            if self.accept_op("("):
                while not self.token.is_op(")"):
                    args.append(self.parse_expression())
                    if not self.accept_op(","):
                        break
                self.expect_op(")")
            attributes.append(A.Attribute(name, args, line))
        return attributes

    def parse_global_declaration(self):
        attributes = self.parse_attributes()
        token = self.token
        if token.is_keyword("var"):
            node = self.parse_global_var(attributes)
        elif token.is_keyword("override"):
            node = self.parse_override(attributes)
        elif token.is_keyword("fn"):
            return self.parse_function(attributes)
        elif attributes:
            raise self.error(f"attributes cannot precede '{token.text}'")
        elif token.is_keyword("const"):
            node = self.parse_const()
        elif token.is_keyword("alias"):
            line = self.advance().line
            name = self.expect_ident()
            self.expect_op("=")
            node = A.Alias(name, self.parse_type(), line)
        elif token.is_keyword("struct"):
            return self.parse_struct()
        elif token.is_keyword("const_assert"):
            line = self.advance().line
            node = A.ConstAssert(self.parse_expression(), line)
        else:
            raise self.error(
                f"expected a module-scope declaration, found '{token.text}'"
            )
        self.expect_op(";")
        return node

    def parse_optionally_typed_ident(self):
        name = self.expect_ident()
        type_ = self.parse_type() if self.accept_op(":") else None
        return name, type_

    def parse_var_template(self):
        if self.token.kind != "template_start":
            return None
        return self.parse_template_list()

    def parse_global_var(self, attributes) -> A.GlobalVar:
        line = self.expect_keyword("var").line
        template_args = self.parse_var_template()
        name, type_ = self.parse_optionally_typed_ident()
        init = self.parse_expression() if self.accept_op("=") else None
        return A.GlobalVar(name, type_, init, template_args, attributes, line)

    def parse_override(self, attributes) -> A.Override:
        line = self.expect_keyword("override").line
        name, type_ = self.parse_optionally_typed_ident()
        init = self.parse_expression() if self.accept_op("=") else None
        return A.Override(name, type_, init, attributes, line)

    def parse_const(self) -> A.Const:
        line = self.expect_keyword("const").line
        name, type_ = self.parse_optionally_typed_ident()
        self.expect_op("=")
        return A.Const(name, type_, self.parse_expression(), line)

    def parse_struct(self) -> A.Struct:
        line = self.expect_keyword("struct").line
        name = self.expect_ident()
        self.expect_op("{")
        members = []
        while not self.token.is_op("}"):
            attributes = self.parse_attributes()
            member_line = self.token.line
            member = self.expect_ident()
            self.expect_op(":")
            members.append(
                A.StructMember(member, self.parse_type(), attributes, member_line)
            )
            if not self.accept_op(","):
                break
        self.expect_op("}")
        return A.Struct(name, members, line)

    def parse_function(self, attributes) -> A.Function:
        line = self.expect_keyword("fn").line
        name = self.expect_ident()
        self.expect_op("(")
        params = []
        while not self.token.is_op(")"):
            param_attributes = self.parse_attributes()
            param_line = self.token.line
            param = self.expect_ident()
            self.expect_op(":")
            params.append(
                A.Param(param, self.parse_type(), param_attributes, param_line)
            )
            if not self.accept_op(","):
                break
        self.expect_op(")")
        return_type, return_attributes = None, []
        if self.accept_op("->"):
            return_attributes = self.parse_attributes()
            return_type = self.parse_type()
        body = self.parse_compound()
        return A.Function(
            name, params, return_type, body, attributes, return_attributes, line
        )

    # Types and template lists.

    def parse_template_list(self) -> list:
        if self.token.kind != "template_start":
            raise self.error("expected a template list")
        self.advance()
        args = []
        while self.token.kind != "template_end":
            args.append(self.parse_expression())
            if not self.accept_op(","):
                break
        if self.token.kind != "template_end":
            raise self.error("unterminated template list")
        self.advance()
        return args

    def parse_type(self) -> A.Ident:
        line = self.token.line
        name = self.expect_ident()
        args = (
            self.parse_template_list() if self.token.kind == "template_start" else None
        )
        return A.Ident(name, args, line)

    # Statements.

    def parse_compound(self) -> A.Block:
        self.parse_attributes()
        line = self.expect_op("{").line
        statements = []
        while not self.token.is_op("}"):
            if self.token.kind == "eof":
                raise self.error("unterminated block")
            statement = self.parse_statement()
            if statement is not None:
                statements.append(statement)
        self.expect_op("}")
        return A.Block(statements, line)

    def parse_statement(self):
        attributes = self.parse_attributes()
        token = self.token
        if attributes and not (
            token.is_op("{") or token.is_keyword(*_STATEMENT_KEYWORDS)
        ):
            raise self.error(f"attributes cannot precede '{token.text}'")
        if token.is_op(";"):
            self.advance()
            return None
        if token.is_op("{"):
            return self.parse_compound()
        if token.is_keyword("if"):
            return self.parse_if()
        if token.is_keyword("switch"):
            return self.parse_switch()
        if token.is_keyword("loop"):
            return self.parse_loop()
        if token.is_keyword("for"):
            return self.parse_for()
        if token.is_keyword("while"):
            line = self.advance().line
            condition = self.parse_expression()
            return A.While(condition, self.parse_compound(), line)
        statement = self.parse_simple_statement()
        self.expect_op(";")
        return statement

    def parse_simple_statement(self):
        """A statement that ends with ``;``, without it."""
        token = self.token
        line = token.line
        if token.is_keyword("return"):
            self.advance()
            value = None if self.token.is_op(";") else self.parse_expression()
            return A.Return(value, line)
        if token.is_keyword("break"):
            self.advance()
            return A.Break(line)
        if token.is_keyword("continue"):
            self.advance()
            return A.Continue(line)
        if token.is_keyword("discard"):
            self.advance()
            return A.Discard(line)
        if token.is_keyword("const_assert"):
            self.advance()
            return A.ConstAssert(self.parse_expression(), line)
        if token.is_keyword("var", "let", "const"):
            return self.parse_local()
        return self.parse_updating_or_call()

    def parse_local(self) -> A.Local:
        token = self.advance()
        template_args = self.parse_var_template() if token.text == "var" else None
        name, type_ = self.parse_optionally_typed_ident()
        init = None
        if token.text != "var" or self.token.is_op("="):
            self.expect_op("=")
            init = self.parse_expression()
        return A.Local(token.text, name, type_, init, template_args, token.line)

    def parse_updating_or_call(self):
        """An assignment, increment, decrement or function call statement."""
        line = self.token.line
        if (
            self.token.kind == "ident"
            and self.token.text == "_"
            and self.peek().is_op("=")
        ):
            self.advance()
            self.advance()
            return A.Assign(None, "=", self.parse_expression(), line)
        target = self.parse_unary()
        if self.token.is_op("++", "--"):
            return A.Increment(target, self.advance().text, line)
        if self.token.is_op("=", *_COMPOUND_ASSIGNMENTS):
            op = self.advance().text
            return A.Assign(target, op, self.parse_expression(), line)
        if isinstance(target, A.Call):
            return A.CallStatement(target, line)
        raise self.error("expected an assignment, an increment or a function call")

    def parse_if(self) -> A.If:
        line = self.expect_keyword("if").line
        condition = self.parse_expression()
        then = self.parse_compound()
        otherwise = None
        if self.token.is_keyword("else"):
            self.advance()
            if self.token.is_keyword("if"):
                otherwise = self.parse_if()
            else:
                otherwise = self.parse_compound()
        return A.If(condition, then, otherwise, line)

    def parse_switch(self) -> A.Switch:
        line = self.expect_keyword("switch").line
        selector = self.parse_expression()
        self.parse_attributes()
        self.expect_op("{")
        clauses = []
        while not self.token.is_op("}"):
            clause_line = self.token.line
            if self.token.is_keyword("default"):
                self.advance()
                selectors = ["default"]
            else:
                self.expect_keyword("case")
                selectors = []
                while not self.token.is_op(":", "{", "@"):
                    if self.token.is_keyword("default"):
                        self.advance()
                        selectors.append("default")
                    else:
                        selectors.append(self.parse_expression())
                    if not self.accept_op(","):
                        break
            self.accept_op(":")
            clauses.append(
                A.SwitchClause(selectors, self.parse_compound(), clause_line)
            )
        self.expect_op("}")
        return A.Switch(selector, clauses, line)

    def parse_loop(self) -> A.Loop:
        line = self.expect_keyword("loop").line
        self.parse_attributes()
        self.expect_op("{")
        statements, continuing, break_if = [], None, None
        while not self.token.is_op("}"):
            if self.token.is_keyword("continuing"):
                continuing, break_if = self.parse_continuing()
                break
            statement = self.parse_statement()
            if statement is not None:
                statements.append(statement)
        self.expect_op("}")
        return A.Loop(A.Block(statements, line), continuing, break_if, line)

    def parse_continuing(self):
        line = self.expect_keyword("continuing").line
        self.parse_attributes()
        self.expect_op("{")
        statements, break_if = [], None
        while not self.token.is_op("}"):
            if self.token.is_keyword("break") and self.peek().is_keyword("if"):
                break_line = self.advance().line
                self.advance()
                break_if = A.BreakIf(self.parse_expression(), break_line)
                self.expect_op(";")
                break
            statement = self.parse_statement()
            if statement is not None:
                statements.append(statement)
        self.expect_op("}")
        return A.Block(statements, line), break_if

    def parse_for(self) -> A.For:
        line = self.expect_keyword("for").line
        self.expect_op("(")
        init = None if self.token.is_op(";") else self.parse_simple_statement()
        self.expect_op(";")
        condition = None if self.token.is_op(";") else self.parse_expression()
        self.expect_op(";")
        update = None if self.token.is_op(")") else self.parse_updating_or_call()
        self.expect_op(")")
        return A.For(init, condition, update, self.parse_compound(), line)

    # Expressions.

    def parse_expression(self, level: int = 0):
        if level == len(_BINARY_LEVELS):
            return self.parse_unary()
        left = self.parse_expression(level + 1)
        operators = _BINARY_LEVELS[level]
        while self.token.kind == "op" and self.token.text in operators:
            token = self.advance()
            right = self.parse_expression(level + 1)
            left = A.Binary(token.text, left, right, token.line)
        return left

    def parse_unary(self):
        token = self.token
        if token.is_op("-", "!", "~", "*", "&"):
            self.advance()
            return A.Unary(token.text, self.parse_unary(), token.line)
        return self.parse_postfix(self.parse_primary())

    def parse_primary(self):
        token = self.token
        if token.kind in ("int", "float"):
            self.advance()
            return A.Literal(token.kind, token.text, token.line)
        if token.is_keyword("true", "false"):
            self.advance()
            return A.Literal("bool", token.text, token.line)
        if token.is_op("("):
            self.advance()
            inner = self.parse_expression()
            self.expect_op(")")
            return inner
        if token.kind == "ident":
            self.advance()
            args = (
                self.parse_template_list()
                if self.token.kind == "template_start"
                else None
            )
            ident = A.Ident(token.text, args, token.line)
            if self.token.is_op("("):
                return A.Call(ident, self.parse_arguments(), token.line)
            return ident
        raise self.error(
            f"expected an expression, found '{token.text or 'end of file'}'"
        )

    def parse_arguments(self) -> list:
        self.expect_op("(")
        args = []
        while not self.token.is_op(")"):
            args.append(self.parse_expression())
            if not self.accept_op(","):
                break
        self.expect_op(")")
        return args

    def parse_postfix(self, expression):
        while True:
            token = self.token
            if token.is_op("["):
                self.advance()
                index = self.parse_expression()
                self.expect_op("]")
                expression = A.Index(expression, index, token.line)
            elif token.is_op("."):
                self.advance()
                expression = A.Member(expression, self.expect_ident(), token.line)
            else:
                return expression

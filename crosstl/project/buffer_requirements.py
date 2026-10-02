"""Bounded, mandatory-prefix buffer footprint analysis for reflected source."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from crosstl.backend import common_ast as ast
from crosstl.backend.DirectX.DirectxAst import IntegerLiteral
from crosstl.project.integral_literals import (
    CFamilyIntegralLiteralError,
    parse_c_family_integral_literal,
)

MAX_BINDING_SIZE = (1 << 63) - 1
_MAX_INDEX = (1 << 31) - 1
_INTEGER_TYPES = {"int", "uint", "int64_t", "uint64_t"}


@dataclass(frozen=True)
class _Buffer:
    name: str


class _StopAnalysis(Exception):
    """The remaining control flow or value identity is not proven."""


def source_buffer_minimum_elements(
    source: str,
    *,
    target: str,
    entry_points: Sequence[str],
    resource_names: Mapping[str, str],
) -> dict[str, int]:
    """Return proven minimum element counts, not complete memory-safety bounds.

    Only a single entry and its mandatory straight-line call prefixes are analyzed.
    A missing result means unknown, not zero. Parameter array extents do not imply
    a footprint. Unsupported parsing, control flow and overloads add no claims.
    """
    if len(entry_points) != 1 or not resource_names:
        return {}
    if re.search(r"^\s*#\s*(?!version\b|extension\b|line\b)\w+", source, re.MULTILINE):
        return {}
    try:
        if target == "directx":
            from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
            from crosstl.backend.DirectX.DirectxParser import HLSLParser

            tree = HLSLParser(HLSLLexer(source).tokenize()).parse()
        elif target == "opengl":
            from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
            from crosstl.backend.GLSL.OpenglParser import GLSLParser

            tree = GLSLParser(
                GLSLLexer(source).tokenize(), shader_type="compute"
            ).parse()
        else:
            return {}
    except (SyntaxError, ValueError, TypeError, IndexError, RecursionError):
        return {}
    analyzer = _FootprintAnalyzer(tree, resource_names)
    try:
        analyzer.call(entry_points[0], [], ())
    except (_StopAnalysis, RecursionError):
        pass
    return analyzer.requirements


def valid_minimum_binding_size(value: Any) -> bool:
    return type(value) is int and 0 < value <= MAX_BINDING_SIZE


def _name(node: Any) -> str | None:
    if isinstance(node, str):
        return node
    if isinstance(node, ast.VariableNode) and not node.vtype:
        return node.name
    if isinstance(node, ast.MemberAccessNode):
        base = _name(node.object)
        if base is not None:
            return f"{base}.{node.member}"
    return None


def _integer(value: Any) -> int | None:
    if type(value) is int:
        result = value
    elif isinstance(value, str):
        try:
            result = parse_c_family_integral_literal(value)
        except CFamilyIntegralLiteralError:
            return None
    else:
        return None
    return result if 0 <= result <= _MAX_INDEX else None


class _FootprintAnalyzer:
    def __init__(self, tree: Any, resources: Mapping[str, str]):
        self.functions: dict[str, list[Any]] = {}
        for function in tree.functions:
            if not getattr(function, "is_prototype", False):
                self.functions.setdefault(function.name, []).append(function)
        self.globals = {name: _Buffer(binding) for name, binding in resources.items()}
        self.requirements: dict[str, int] = {}
        self.remaining = 20000

    def call(self, name, arguments, stack):
        functions = self.functions.get(name, ())
        if len(functions) != 1 or name in stack or len(stack) >= 32:
            raise _StopAnalysis
        function = functions[0]
        # Entry system-value parameters carry no constant facts.
        if not stack and not arguments:
            arguments = [None] * len(function.params)
        if len(arguments) != len(function.params):
            raise _StopAnalysis
        environment = dict(self.globals)
        for parameter, value in zip(function.params, arguments):
            if set(parameter.qualifiers) & {"out", "inout"} or "&" in parameter.vtype:
                raise _StopAnalysis
            if type(value) is int and parameter.vtype not in _INTEGER_TYPES:
                value = None
            self.bind(environment, parameter.name, value)
        stack = (*stack, name)
        for statement in function.body:
            if isinstance(statement, ast.ReturnNode):
                self.check_expression(statement.value, environment)
                value = self.expression(statement.value, environment, stack)
                return value if function.return_type in _INTEGER_TYPES else None
            if isinstance(statement, ast.VariableNode) and statement.vtype:
                # The declarator is in scope in its initializer; do not retain
                # a same-named global resource when a local shadows it.
                self.bind(environment, statement.name, None)
                self.check_expression(statement.value, environment)
                value = self.expression(statement.value, environment, stack)
                if type(value) is int and statement.vtype not in _INTEGER_TYPES:
                    value = None
                self.bind(environment, statement.name, value)
            elif isinstance(statement, ast.AssignmentNode):
                self.check_expression(statement.left, environment)
                self.check_expression(statement.right, environment)
                self.expression(statement.left, environment, stack)
                self.expression(statement.right, environment, stack)
                name = _name(statement.left)
                if name is not None:
                    self.bind(environment, name, None)
            elif isinstance(statement, (ast.FunctionCallNode, ast.UnaryOpNode)):
                self.check_expression(statement, environment)
                self.expression(statement, environment, stack)
            else:
                raise _StopAnalysis
        return None

    @staticmethod
    def bind(environment, name, value):
        for key in tuple(environment):
            if key.startswith(f"{name}."):
                del environment[key]
        environment[name] = value

    def access(self, buffer, index):
        if isinstance(buffer, _Buffer) and type(index) is int and index >= 0:
            self.requirements[buffer.name] = max(
                self.requirements.get(buffer.name, 0), index + 1
            )

    def check_expression(self, expression, environment):
        # Reject potential argument mutations before choosing an evaluation
        # order or using constant facts from any sibling expression.
        pending = [expression]
        while pending:
            self.remaining -= 1
            if self.remaining < 0:
                raise _StopAnalysis
            node = pending.pop()
            if isinstance(node, (ast.AssignmentNode, ast.PostfixOpNode)) or (
                isinstance(node, ast.UnaryOpNode) and node.op in {"++", "--"}
            ):
                raise _StopAnalysis
            if isinstance(node, ast.FunctionCallNode):
                name = _name(node.name)
                if name in self.functions:
                    functions = self.functions[name]
                    if len(functions) != 1 or any(
                        set(param.qualifiers) & {"out", "inout"} or "&" in param.vtype
                        for param in functions[0].params
                    ):
                        raise _StopAnalysis
                elif not self.is_constructor(name):
                    method = node.name
                    if not (
                        isinstance(method, ast.MemberAccessNode)
                        and method.member == "Load"
                        and isinstance(environment.get(_name(method.object)), _Buffer)
                        and len(node.args) == 1
                    ):
                        raise _StopAnalysis
            if isinstance(node, ast.ASTNode):
                for value in vars(node).values():
                    if isinstance(value, ast.ASTNode):
                        pending.append(value)
                    elif isinstance(value, (list, tuple)):
                        pending.extend(value)

    @staticmethod
    def is_constructor(name):
        return name in _INTEGER_TYPES | {"float", "double", "bool"} or (
            name is not None and re.fullmatch(r"[iubd]?vec[234]", name) is not None
        )

    def expression(self, node, environment, stack):
        self.remaining -= 1
        if self.remaining < 0:
            raise _StopAnalysis
        if node is None:
            return None
        if type(node) in (float, bool):
            return None
        name = _name(node)
        if name is not None and name in environment:
            return environment[name]
        if type(node) in (int, str):
            return _integer(node)
        if isinstance(node, IntegerLiteral):
            return _integer(int(node))
        if type(node).__name__ == "NumberNode":
            return _integer(node.value)
        if isinstance(node, ast.VariableNode):
            return None
        if isinstance(node, ast.MemberAccessNode):
            self.expression(node.object, environment, stack)
            return None
        if isinstance(node, ast.ArrayAccessNode):
            buffer = self.expression(node.array, environment, stack)
            index = self.expression(node.index, environment, stack)
            self.access(buffer, index)
            return None
        if isinstance(node, ast.BinaryOpNode):
            left = self.expression(node.left, environment, stack)
            if node.op in {"&&", "||", "AND", "OR"}:
                raise _StopAnalysis
            right = self.expression(node.right, environment, stack)
            if type(left) is not int or type(right) is not int:
                return None
            if node.op == "+":
                return _integer(left + right)
            if node.op == "-":
                return _integer(left - right)
            if node.op == "*":
                return _integer(left * right)
            return None
        if isinstance(node, ast.UnaryOpNode):
            value = self.expression(node.operand, environment, stack)
            if node.op in {"++", "--"}:
                raise _StopAnalysis
            if type(value) is int and node.op in {"+", "-"}:
                return _integer(value if node.op == "+" else -value)
            return None
        if isinstance(node, ast.VectorConstructorNode):
            arguments = [self.expression(arg, environment, stack) for arg in node.args]
            return self.integer_cast(node.vector_type, arguments)
        if isinstance(node, ast.FunctionCallNode):
            name = _name(node.name)
            arguments = [self.expression(arg, environment, stack) for arg in node.args]
            if name in self.functions:
                return self.call(name, arguments, stack)
            if name in _INTEGER_TYPES:
                return self.integer_cast(name, arguments)
            if (
                isinstance(node.name, ast.MemberAccessNode)
                and node.name.member == "Load"
            ):
                buffer = self.expression(node.name.object, environment, stack)
                if isinstance(buffer, _Buffer) and len(arguments) == 1:
                    self.access(buffer, arguments[0])
                    return None
            # GLSL constructors do not mutate their scalar arguments. Other
            # unresolved calls may write through out parameters: stop here.
            if self.is_constructor(name):
                return None
            raise _StopAnalysis
        raise _StopAnalysis

    @staticmethod
    def integer_cast(name, arguments):
        if name not in _INTEGER_TYPES or len(arguments) != 1:
            return None
        value = arguments[0]
        if type(value) is not int or (name.startswith("u") and value < 0):
            return None
        return _integer(value)

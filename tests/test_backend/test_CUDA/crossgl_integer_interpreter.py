"""A strict interpreter for the integer subset of CrossGL that the directed-rounding functions are written in.

Floating-point values are carried as their bit patterns: the functions only move them and reinterpret their bits.
The interpreter rejects what a C-family or shading-language compiler could treat differently: a binary operator on
operands of different types, a shift by a negative amount or by at least the operand width, and signed overflow.
"""

from __future__ import annotations

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

WIDTHS = {"u32": 32, "i32": 32, "u64": 64, "i64": 64, "f32": 32, "f64": 64}
LITERAL_TYPES = {"uint": "u32", "int": "i32", "bool": "bool"}


class InterpreterError(AssertionError):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


def _wrap(kind, value):
    if kind == "bool":
        return bool(value)
    bits = WIDTHS[kind]
    value &= (1 << bits) - 1
    if kind in ("i32", "i64") and value >> (bits - 1):
        value -= 1 << bits
    return value


def _checked_signed(kind, value):
    bits = WIDTHS[kind]
    if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise InterpreterError(f"signed overflow: {value} does not fit {kind}")
    return value


class Interpreter:
    def __init__(self, source):
        program = Parser(Lexer(source).tokens).parse()
        self.functions = {function.name: function for function in program.functions}
        self.steps = 0

    def call(self, name, *arguments):
        """Run ``name`` on ``(type, value)`` arguments and return its ``(type, value)`` result."""
        return self.call_with_outputs(name, *arguments)[0]

    def call_with_outputs(self, name, *arguments):
        """Run ``name``; return its result and a dict of the values of its ``out`` parameters."""
        function = self.functions[name]
        frame = [{}]
        for parameter, argument in zip(function.parameters, arguments):
            declared = parameter.param_type.name
            if argument[0] != declared:
                raise InterpreterError(
                    f"{name}: argument of type {argument[0]} for {declared} {parameter.name}"
                )
            frame[0][parameter.name] = list(argument)
        result = None
        try:
            self.block(function.body.statements, frame)
        except _Return as returned:
            declared = function.return_type.name
            if returned.value[0] != declared:
                raise InterpreterError(
                    f"{name} returns {returned.value[0]}, declared {declared}"
                )
            result = returned.value
        outputs = {
            parameter.name: frame[0][parameter.name][1]
            for parameter in function.parameters
            if "out" in (parameter.qualifiers or [])
        }
        return result, outputs

    # statements

    def block(self, statements, frame):
        frame.append({})
        try:
            for statement in statements:
                self.statement(statement, frame)
        finally:
            frame.pop()

    def statement(self, node, frame):
        kind = type(node).__name__
        self.steps += 1
        if kind == "VariableNode":
            declared = node.var_type.name
            value = (declared, 0 if declared != "bool" else False)
            if node.initial_value is not None:
                value = self.expression(node.initial_value, frame)
                if value[0] != declared:
                    raise InterpreterError(
                        f"{node.name}: {value[0]} initializer for {declared}"
                    )
            frame[-1][node.name] = list(value)
        elif kind == "ExpressionStatementNode":
            self.expression(node.expression, frame)
        elif kind == "ReturnNode":
            raise _Return(self.expression(node.value, frame))
        elif kind == "IfNode":
            if self.condition(node.condition, frame):
                self.branch(node.then_branch, frame)
            elif node.else_branch is not None:
                self.branch(node.else_branch, frame)
        elif kind == "ForNode":
            frame.append({})
            try:
                self.statement(node.init, frame)
                while self.condition(node.condition, frame):
                    self.branch(node.body, frame)
                    self.expression(node.update, frame)
            finally:
                frame.pop()
        elif kind == "BlockNode":
            self.block(node.statements, frame)
        else:
            raise InterpreterError(f"unsupported statement {kind}")

    def branch(self, node, frame):
        if type(node).__name__ == "BlockNode":
            self.block(node.statements, frame)
        else:
            self.statement(node, frame)

    def condition(self, node, frame):
        value = self.expression(node, frame)
        if value[0] != "bool":
            raise InterpreterError(f"condition of type {value[0]}")
        return value[1]

    # expressions

    def lookup(self, name, frame):
        for scope in reversed(frame):
            if name in scope:
                return scope[name]
        raise InterpreterError(f"undeclared {name}")

    def expression(self, node, frame):
        kind = type(node).__name__
        if kind == "LiteralNode":
            literal = node.literal_type.name
            if literal not in LITERAL_TYPES:
                raise InterpreterError(f"unsupported literal type {literal}")
            return (LITERAL_TYPES[literal], node.value)
        if kind == "IdentifierNode":
            if node.name in ("true", "false"):
                return ("bool", node.name == "true")
            return tuple(self.lookup(node.name, frame))
        if kind == "AssignmentNode":
            if node.operator != "=" or type(node.target).__name__ != "IdentifierNode":
                raise InterpreterError("only plain assignment to a variable")
            slot = self.lookup(node.target.name, frame)
            value = self.expression(node.value, frame)
            if value[0] != slot[0]:
                raise InterpreterError(
                    f"assigning {value[0]} to {slot[0]} {node.target.name}"
                )
            slot[1] = value[1]
            return value
        if kind == "TernaryOpNode":
            chosen = (
                node.true_expr
                if self.condition(node.condition, frame)
                else node.false_expr
            )
            return self.expression(chosen, frame)
        if kind == "UnaryOpNode":
            return self.unary(node, frame)
        if kind == "BinaryOpNode":
            return self.binary(node, frame)
        if kind == "FunctionCallNode":
            return self.function_call(node, frame)
        raise InterpreterError(f"unsupported expression {kind}")

    def unary(self, node, frame):
        operator = node.operator
        if operator == "++":
            slot = self.lookup(node.operand.name, frame)
            slot[1] = (
                _checked_signed(slot[0], slot[1] + 1)
                if slot[0][0] == "i"
                else _wrap(slot[0], slot[1] + 1)
            )
            return tuple(slot)
        kind, value = self.expression(node.operand, frame)
        if operator == "!":
            if kind != "bool":
                raise InterpreterError("! on a non-bool")
            return ("bool", not value)
        if operator == "~" and kind in ("u32", "u64"):
            return (kind, _wrap(kind, ~value))
        if operator == "-" and kind in ("i32", "i64"):
            return (kind, _checked_signed(kind, -value))
        raise InterpreterError(f"unsupported unary {operator} on {kind}")

    def binary(self, node, frame):
        operator = node.operator
        left = self.expression(node.left, frame)
        if operator in ("&&", "||"):
            if left[0] != "bool":
                raise InterpreterError(f"{operator} on {left[0]}")
            if (operator == "&&") != left[1]:
                return ("bool", left[1])
            right = self.expression(node.right, frame)
            if right[0] != "bool":
                raise InterpreterError(f"{operator} on {right[0]}")
            return ("bool", right[1])
        right = self.expression(node.right, frame)
        kind, a = left
        if operator in ("<<", ">>"):
            if kind not in ("u32", "u64") or right[0] not in ("u32", "i32"):
                raise InterpreterError(f"shift of {kind} by {right[0]}")
            amount = right[1]
            if not 0 <= amount < WIDTHS[kind]:
                raise InterpreterError(f"shift of {kind} by {amount}")
            return (kind, _wrap(kind, a << amount if operator == "<<" else a >> amount))
        if right[0] != kind:
            raise InterpreterError(f"{kind} {operator} {right[0]}")
        b = right[1]
        if operator in ("==", "!=", "<", ">", "<=", ">="):
            if kind == "bool" and operator not in ("==", "!="):
                raise InterpreterError(f"bool {operator}")
            result = {
                "==": a == b,
                "!=": a != b,
                "<": a < b,
                ">": a > b,
                "<=": a <= b,
                ">=": a >= b,
            }[operator]
            return ("bool", result)
        if kind == "bool":
            raise InterpreterError(f"bool {operator}")
        if operator == "/":
            if b == 0:
                raise InterpreterError("division by zero")
            quotient = abs(a) // abs(b)
            result = quotient if (a < 0) == (b < 0) else -quotient
        else:
            operations = {
                "+": lambda: a + b,
                "-": lambda: a - b,
                "*": lambda: a * b,
                "&": lambda: a & b,
                "|": lambda: a | b,
                "^": lambda: a ^ b,
            }
            if operator not in operations:
                raise InterpreterError(f"unsupported operator {operator}")
            result = operations[operator]()
        if kind in ("i32", "i64"):
            return (kind, _checked_signed(kind, result))
        return (kind, _wrap(kind, result))

    def function_call(self, node, frame):
        name = node.function.name
        if name in self.functions:
            function = self.functions[name]
            arguments = [
                self.expression(argument, frame) for argument in node.arguments
            ]
            result_slots = []
            for parameter, argument_node in zip(function.parameters, node.arguments):
                if "out" in (parameter.qualifiers or []):
                    if type(argument_node).__name__ != "IdentifierNode":
                        raise InterpreterError("out argument is not a variable")
                    result_slots.append(
                        (parameter.name, self.lookup(argument_node.name, frame))
                    )
            callee = [{}]
            for parameter, argument in zip(function.parameters, arguments):
                if argument[0] != parameter.param_type.name:
                    raise InterpreterError(
                        f"{name}: {argument[0]} for {parameter.param_type.name}"
                    )
                callee[0][parameter.name] = list(argument)
            result = None
            try:
                self.block(function.body.statements, callee)
            except _Return as returned:
                result = returned.value
                if result[0] != function.return_type.name:
                    raise InterpreterError(f"{name} returns {result[0]}")
            for parameter_name, slot in result_slots:
                slot[1] = callee[0][parameter_name][1]
            return result
        arguments = [self.expression(argument, frame) for argument in node.arguments]
        return self.builtin(name, arguments)

    def builtin(self, name, arguments):
        (kind, value), *rest = arguments
        if rest:
            raise InterpreterError(f"{name} with {len(arguments)} arguments")
        if name in ("u32", "i32", "u64", "i64"):
            if kind not in ("u32", "i32", "u64", "i64"):
                raise InterpreterError(f"{name}({kind})")
            return (name, _wrap(name, value))
        casts = {
            "floatBitsToUint": ("f32", "u32"),
            "uintBitsToFloat": ("u32", "f32"),
            "as_type<u64>": ("f64", "u64"),
            "as_type<f64>": ("u64", "f64"),
        }
        if name in casts:
            source, target = casts[name]
            if kind != source:
                raise InterpreterError(f"{name}({kind})")
            return (target, value)
        if name == "findMSB" and kind == "u32":
            return ("i32", value.bit_length() - 1)
        raise InterpreterError(f"unsupported call {name}({kind})")

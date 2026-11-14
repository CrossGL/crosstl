"""Calls of program functions in compile-time initializers, evaluated at translation time.

MSL and HLSL let a program-scope constant call a function of the program: ``constant constexpr int C = twice(base);``
after ``constexpr int twice(int x) { return 2 * x; }``. The compiler evaluates the call (C++17 [expr.const]/2 and
[dcl.constexpr]/9). CrossGL keeps the call in the constant's initializer, but most targets cannot. A GLSL constant
expression calls built-in functions only (GLSL 4.60 section 4.3.3), and a WGSL const-expression likewise. The GLSL,
HLSL and SPIR-V generators therefore refused such a constant. The MSL, Slang, WGSL and CUDA generators printed the
call ahead of the function it names, and their compilers reject that.

:func:`fold_constant_calls` evaluates, for every target, each call of a program function in the initializer of a
``constant`` or ``const`` variable at shader or stage scope, in the value of a CrossGL constant declaration and in
the condition of a ``static_assert``, and replaces the call by its value. The value is spelled as a literal of the
function's return type, or as a conversion of a literal for the integer types that have no literal suffix.

The evaluator runs the function body: local scalar variables, assignments, ``++`` and ``--``, ``if``, ``for``,
``while``, ``do``, ``break``, ``continue`` and ``return``, calls of program functions, conversions, and ``min``,
``max``, ``abs`` and ``clamp`` of scalars. It reads parameters, locals and the program's scalar constants. Values
are ``bool``, the integer types of 32 and 64 bits and ``float``, and follow the source languages exactly:

* Binary operators apply the usual arithmetic conversions (C++17 [expr]/11). Unsigned results wrap modulo ``2**N``
  ([basic.fundamental]/4). A signed overflow, a division by zero, a shift count outside ``[0, N)`` or a left shift
  of a negative value is not a constant expression ([expr.const]/2.6), so the call stays.
* ``float`` arithmetic is IEEE 754 binary32 with every operation rounded to nearest even, which is how the source
  compilers evaluate constant expressions. A result that overflows or is not a number leaves the call in place.
* A conversion to an integer type wraps modulo ``2**N`` (:mod:`.integral_types`). A conversion of a ``float`` that
  the integer type cannot represent is undefined ([conv.fpint]/1), and the call stays.

Integers narrower than 32 bits may be converted and stored but not computed with: C++ promotes them to ``int``,
while GLSL and WGSL compute in the narrow type, so their arithmetic has no single meaning in CrossGL. ``half`` and
``double`` values, vectors, arrays, structures, references, ``out`` parameters, specialization constants and any
other construct leave the call in place too, as do evaluations past a step or depth limit. A call that stays
reaches the generators as before, which refuse it where the target cannot express it.
"""

from __future__ import annotations

import math
import struct
from typing import NamedTuple

from .. import ast as A
from .hub_types import (
    BOOL,
    FLOAT32,
    INT32,
    UINT32,
    HubType,
    arithmetic_result,
    call_name,
    is_pure_callee,
    promote,
    type_from_name,
    type_from_node,
)
from .integral_types import ProgramTypeAliases
from .rewrite import child_slots, get_child, set_child
from .scoped_walk import program_stages

__all__ = ["fold_constant_calls"]

_STEP_LIMIT = 200_000
_DEPTH_LIMIT = 64
_COMPILE_TIME_QUALIFIERS = frozenset({"constant", "const"})
_CONSTANT_QUALIFIERS = _COMPILE_TIME_QUALIFIERS | {"static", "constexpr"}
_LOCAL_QUALIFIERS = frozenset({"const", "constant", "constexpr"})
_BUILTIN_ARITIES = {"abs": 1, "min": 2, "max": 2, "clamp": 3}
_COMPARISONS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}
_INTEGER_OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
}
_FLOAT_OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}
_COMPOUND_OPERATORS = frozenset(
    {"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
)
# CrossGL spellings of the integer types whose literals have no suffix.
_CONVERSION_SPELLINGS = {
    HubType("i", 8): "int8_t",
    HubType("u", 8): "uint8_t",
    HubType("i", 16): "int16_t",
    HubType("u", 16): "uint16_t",
    HubType("i", 64): "int64_t",
    HubType("u", 64): "uint64_t",
}
_AMBIGUOUS = object()
_UNINITIALIZED = object()


class _Unresolved(Exception):
    """The evaluator does not compute this construct; the call it belongs to stays in place."""


class _Value(NamedTuple):
    type: HubType
    value: object


def _supported(hub):
    if hub is not None and hub.lanes == 1:
        if hub.kind == "b" or (hub.kind == "f" and hub.bits == 32):
            return hub
        if hub.kind in ("i", "u") and hub.bits in (8, 16, 32, 64):
            return hub
    raise _Unresolved(f"values of type {hub} are not evaluated")


def _is_narrow(hub):
    return hub.kind in ("i", "u") and hub.bits < 32


def _bounds(hub):
    if hub.kind == "u":
        return 0, (1 << hub.bits) - 1
    return -(1 << (hub.bits - 1)), (1 << (hub.bits - 1)) - 1


def _wrap(value, hub):
    modulus = 1 << hub.bits
    value %= modulus
    if hub.kind == "i" and value >= modulus >> 1:
        value -= modulus
    return value


def _binary32(number):
    """``number`` rounded to the nearest binary32 value, ties to even; overflow and NaN are unresolved."""
    try:
        rounded = struct.unpack("<f", struct.pack("<f", number))[0]
    except OverflowError:
        raise _Unresolved("the value overflows float") from None
    if not math.isfinite(rounded):
        raise _Unresolved("the value is not a finite float")
    return rounded


def _binary32_literal(number):
    """The binary32 value of a ``float`` literal that the AST holds as the binary64 nearest to its decimal text.

    Rounding the decimal to binary64 and then to binary32 can differ from rounding it to binary32 directly only when
    the binary64 value lies within one of its own units in the last place of a binary32 rounding boundary, so such a
    literal is unresolved.
    """
    rounded = _binary32(number)
    if rounded == number:
        return rounded
    bits = struct.unpack("<I", struct.pack("<f", rounded))[0]
    away = 1 if (number > rounded) == (rounded > 0) else -1
    neighbour = struct.unpack("<f", struct.pack("<I", bits + away))[0]
    if abs(number - (rounded + neighbour) / 2) <= math.ulp(number):
        raise _Unresolved("the literal lies too close to a float rounding boundary")
    return rounded


def _convert(value, to):
    source, raw = value
    if source == to:
        return value
    if to.kind == "b":
        return _Value(BOOL, bool(raw))
    if to.kind == "f":
        if source.kind == "b":
            return _Value(to, 1.0 if raw else 0.0)
        if source.kind in ("i", "u"):
            if abs(raw) >= 1 << 53:
                raise _Unresolved("the integer is too large to convert exactly")
            return _Value(to, _binary32(float(raw)))
        return _Value(to, raw)
    if source.kind == "b":
        return _Value(to, int(raw))
    if source.kind == "f":
        truncated = math.trunc(raw)
        low, high = _bounds(to)
        if not low <= truncated <= high:
            raise _Unresolved("the float is out of the integer type's range")
        return _Value(to, truncated)
    return _Value(to, _wrap(raw, to))


def _arithmetic_type(value):
    if value.type.kind not in ("i", "u", "f") or _is_narrow(value.type):
        raise _Unresolved(f"arithmetic on {value.type} is not evaluated")
    return value.type


def _integer_type(value):
    if value.type.kind not in ("i", "u") or _is_narrow(value.type):
        raise _Unresolved(f"integer operators on {value.type} are not evaluated")
    return value.type


def _integer_operation(operator, a, b, hub):
    if operator in ("/", "%"):
        if b == 0:
            raise _Unresolved("the expression divides by zero")
        quotient = abs(a) // abs(b)
        if (a < 0) != (b < 0):
            quotient = -quotient
        result = quotient if operator == "/" else a - b * quotient
    elif operator in _INTEGER_OPERATIONS:
        result = _INTEGER_OPERATIONS[operator](a, b)
    else:
        raise _Unresolved(f"operator {operator} is not evaluated on integers")
    if hub.kind == "u":
        return _wrap(result, hub)
    low, high = _bounds(hub)
    if not low <= result <= high:
        raise _Unresolved("the signed result overflows")
    return result


def _float_operation(operator, a, b):
    if operator == "/":
        if b == 0.0:
            raise _Unresolved("the expression divides by zero")
        return _binary32(a / b)
    if operator in _FLOAT_OPERATIONS:
        return _binary32(_FLOAT_OPERATIONS[operator](a, b))
    raise _Unresolved(f"operator {operator} is not evaluated on floats")


def _arithmetic(operator, left, right):
    common = arithmetic_result(_arithmetic_type(left), _arithmetic_type(right))
    a = _convert(left, common).value
    b = _convert(right, common).value
    if common.kind == "f":
        return _Value(common, _float_operation(operator, a, b))
    return _Value(common, _integer_operation(operator, a, b, common))


def _shift(operator, left, right):
    hub = promote(_integer_type(left))
    _integer_type(right)
    count, value = right.value, left.value
    if not 0 <= count < hub.bits:
        raise _Unresolved("the shift count is out of range")
    if operator == ">>":
        return _Value(hub, value >> count)
    if hub.kind == "u":
        return _Value(hub, _wrap(value << count, hub))
    if value < 0 or value << count > _bounds(hub)[1]:
        raise _Unresolved("the signed left shift overflows")
    return _Value(hub, value << count)


def _compare(operator, left, right):
    if left.type.kind == "b" and right.type.kind == "b":
        if operator not in ("==", "!="):
            raise _Unresolved("booleans are only compared for equality")
        return _Value(BOOL, _COMPARISONS[operator](left.value, right.value))
    common = arithmetic_result(_arithmetic_type(left), _arithmetic_type(right))
    a = _convert(left, common).value
    b = _convert(right, common).value
    return _Value(BOOL, _COMPARISONS[operator](a, b))


def _apply(operator, left, right):
    if operator in _COMPARISONS:
        return _compare(operator, left, right)
    if operator in ("<<", ">>"):
        return _shift(operator, left, right)
    return _arithmetic(operator, left, right)


def _mentions(node, name):
    if isinstance(node, A.IdentifierNode):
        return node.name == name
    if not isinstance(node, A.ASTNode):
        return False
    return any(
        _mentions(get_child(node, field, key), name) for field, key in child_slots(node)
    )


def _qualifiers(node):
    return {str(qualifier) for qualifier in getattr(node, "qualifiers", None) or []}


def _is_compile_time_variable(node):
    """A ``constant`` or ``const`` variable with an initializer and no other storage: a compile-time value."""
    if not isinstance(node, A.VariableNode) or isinstance(node, A.ArrayNode):
        return False
    if getattr(node, "is_type_alias", False) or node.initial_value is None:
        return False
    if getattr(node, "resource_qualifiers", None):
        return False
    qualifiers = _qualifiers(node)
    return (
        bool(qualifiers & _COMPILE_TIME_QUALIFIERS)
        and qualifiers <= _CONSTANT_QUALIFIERS
    )


def _constant_table(variables, constants):
    """Name -> declaration of the scalar compile-time values an evaluator may read, or ``_AMBIGUOUS``."""
    table = {}
    for node in list(variables or []) + list(constants or []):
        name = getattr(node, "name", None)
        if not isinstance(name, str):
            continue
        if isinstance(node, A.ConstantNode) or _is_compile_time_variable(node):
            # A specialization constant's value is the pipeline's, not its initializer.
            readable = not getattr(node, "attributes", None)
            entry = node if readable else _AMBIGUOUS
        else:
            entry = _AMBIGUOUS
        table[name] = _AMBIGUOUS if name in table else entry
    return table


def _function_table(functions):
    table = {}
    for function in functions or []:
        if isinstance(function, A.FunctionNode) and isinstance(function.name, str):
            table.setdefault(function.name, []).append(function)
    return table


def _declared(entry):
    if isinstance(entry, A.ConstantNode):
        return entry.const_type, entry.value
    return entry.var_type, entry.initial_value


class _Frame:
    """The block scopes of one function invocation: name -> ``[type, value]``."""

    def __init__(self, stage_visible):
        self.scopes = [{}]
        self.stage_visible = stage_visible

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def declare(self, name, hub, value):
        self.scopes[-1][name] = [hub, value]

    def find(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None


class _Evaluator:
    """Evaluates expressions of one scope (the shader, or a stage within it) and the program functions they call."""

    def __init__(self, aliases, shader_scope, stage_scope=None):
        self.aliases = aliases
        self.shader_constants, self.shader_functions = shader_scope
        self.stage_constants, self.stage_functions = stage_scope or ({}, {})
        self.stage_visible = self.scope_stage_visible = stage_scope is not None
        self.values = {}
        self.pending = set()
        self.frame = None
        self.steps = 0
        self.depth = 0

    # Scope lookups -------------------------------------------------------------------------------------------------

    def scalar_type(self, type_node):
        if isinstance(type_node, (A.ReferenceType, A.PointerType, A.ArrayType)):
            raise _Unresolved("references, pointers and arrays are not evaluated")
        return _supported(type_from_node(self.aliases.resolve_type(type_node)))

    def overloads(self, name):
        if self.stage_visible and name in self.stage_functions:
            if name in self.shader_functions:
                raise _Unresolved(f"{name} names functions of several scopes")
            return self.stage_functions[name]
        return self.shader_functions.get(name)

    def is_user_call(self, node):
        try:
            return bool(self.overloads(call_name(node)))
        except _Unresolved:
            return False

    def _step(self):
        self.steps += 1
        if self.steps > _STEP_LIMIT:
            raise _Unresolved("the evaluation exceeds the step limit")

    def _enter(self):
        self.depth += 1
        if self.depth > _DEPTH_LIMIT:
            raise _Unresolved("the evaluation nests too deeply")

    def _constant(self, name):
        tables = [self.shader_constants]
        if self.stage_visible:
            tables.insert(0, self.stage_constants)
        entry = next((table[name] for table in tables if name in table), None)
        if entry is None:
            raise _Unresolved(f"{name} is not a compile-time constant")
        if entry is _AMBIGUOUS:
            raise _Unresolved(f"{name} is not a scalar constant the evaluator may read")
        key = id(entry)
        if key in self.values:
            return self.values[key]
        if key in self.pending:
            raise _Unresolved(f"{name} is defined in terms of itself")
        saved = (self.frame, self.stage_visible)
        self.frame = None
        self.stage_visible = name in self.stage_constants and self.stage_visible
        self.pending.add(key)
        try:
            self._enter()
            declared_type, initializer = _declared(entry)
            value = _convert(
                self.evaluate(initializer), self.scalar_type(declared_type)
            )
        finally:
            self.depth -= 1
            self.pending.discard(key)
            self.frame, self.stage_visible = saved
        self.values[key] = value
        return value

    def _slot(self, name):
        slot = self.frame.find(name) if self.frame is not None else None
        if slot is None:
            raise _Unresolved(
                f"{name} is not a local variable of the evaluated function"
            )
        return slot

    # Expressions ---------------------------------------------------------------------------------------------------

    def fold(self, call):
        """The value of a call of a program function, evaluated from scratch."""
        self.steps = 0
        self.depth = 0
        self.frame, self.stage_visible = None, self.scope_stage_visible
        try:
            return self.evaluate(call)
        except RecursionError:
            raise _Unresolved("the evaluation nests too deeply") from None

    def evaluate(self, node):
        self._step()
        if isinstance(node, A.LiteralNode):
            return self._literal(node)
        if isinstance(node, A.IdentifierNode):
            return self._identifier(node.name)
        if isinstance(node, A.BinaryOpNode):
            return self._binary(str(node.operator), node.left, node.right)
        if isinstance(node, A.UnaryOpNode) and not node.is_postfix:
            return self._unary(str(node.operator), node.operand)
        if isinstance(node, A.TernaryOpNode):
            result_type = self.static_type(node)
            branch = node.true_expr if self._truth(node.condition) else node.false_expr
            return _convert(self.evaluate(branch), result_type)
        if isinstance(node, A.CastNode):
            return _convert(
                self.evaluate(node.expression), self.scalar_type(node.target_type)
            )
        if isinstance(node, A.ConstructorNode) and not node.named_arguments:
            return self._conversion(node.constructor_type, list(node.arguments or []))
        if isinstance(node, A.FunctionCallNode) and not node.generic_args:
            return self._call(node)
        raise _Unresolved(f"{type(node).__name__} is not evaluated")

    def _truth(self, node):
        return _convert(self.evaluate(node), BOOL).value

    def _literal(self, node):
        value = node.value
        declared = node.literal_type
        if isinstance(value, bool):
            return _Value(BOOL, value)
        hub = None if declared is None else self.scalar_type(declared)
        if isinstance(value, int):
            hub = hub or INT32
            if hub.kind == "f":
                return _convert(_Value(INT32, value), hub)
            low, high = _bounds(hub) if hub.kind != "b" else (0, 1)
            if not low <= value <= high:
                raise _Unresolved(f"the literal {value} does not fit {hub}")
            return _Value(hub, bool(value) if hub.kind == "b" else value)
        if isinstance(value, float) and (hub is None or hub == FLOAT32):
            return _Value(FLOAT32, _binary32_literal(value))
        raise _Unresolved(f"the literal {value!r} is not evaluated")

    def _identifier(self, name):
        if name in ("true", "false"):
            return _Value(BOOL, name == "true")
        slot = self.frame.find(name) if self.frame is not None else None
        if slot is None:
            return self._constant(name)
        if slot[1] is _UNINITIALIZED:
            raise _Unresolved(f"{name} is read before it is assigned")
        return _Value(slot[0], slot[1])

    def _binary(self, operator, left_node, right_node):
        if operator in ("&&", "||"):
            left = self._truth(left_node)
            if left == (operator == "||"):
                return _Value(BOOL, left)
            return _Value(BOOL, self._truth(right_node))
        return _apply(operator, self.evaluate(left_node), self.evaluate(right_node))

    def _unary(self, operator, operand_node):
        if operator == "!":
            return _Value(BOOL, not self._truth(operand_node))
        operand = self.evaluate(operand_node)
        if operator == "+":
            return _convert(operand, promote(_arithmetic_type(operand)))
        if operator == "-":
            hub = promote(_arithmetic_type(operand))
            if hub.kind == "f":
                return _Value(hub, -operand.value)
            return _Value(hub, _integer_operation("-", 0, operand.value, hub))
        if operator == "~":
            hub = promote(_integer_type(operand))
            return _Value(hub, _wrap(~operand.value, hub))
        raise _Unresolved(f"operator {operator} is not evaluated")

    def _conversion(self, type_node, arguments):
        if len(arguments) != 1:
            raise _Unresolved("only scalar conversions are evaluated")
        return _convert(self.evaluate(arguments[0]), self.scalar_type(type_node))

    def _call(self, node):
        name = call_name(node)
        arguments = list(node.arguments or [])
        overloads = self.overloads(name)
        if overloads:
            return self._invoke(name, overloads, arguments)
        if type_from_name(self.aliases.resolve(name)) is not None:
            return self._conversion(name, arguments)
        if name in _BUILTIN_ARITIES:
            return self._builtin(name, [self.evaluate(a) for a in arguments])
        raise _Unresolved(f"{name or 'the callee'} is not evaluated")

    def _builtin(self, name, values):
        if len(values) != _BUILTIN_ARITIES[name]:
            raise _Unresolved(f"{name} takes {_BUILTIN_ARITIES[name]} arguments")
        hub = _arithmetic_type(values[0])
        if any(value.type != hub for value in values):
            raise _Unresolved(f"the arguments of {name} differ in type")
        raw = [value.value for value in values]
        zero_signs = {math.copysign(1.0, v) for v in raw if hub.kind == "f" and v == 0}
        if name != "abs" and len(zero_signs) > 1:
            raise _Unresolved(f"{name} of zeros of both signs may return either")
        if name == "abs":
            result = abs(raw[0])
            if hub.kind == "i" and result > _bounds(hub)[1]:
                raise _Unresolved("abs overflows")
        elif name == "clamp":
            if raw[1] > raw[2]:
                raise _Unresolved("clamp with a lower bound above the upper bound")
            result = min(max(raw[0], raw[1]), raw[2])
        else:
            result = min(raw) if name == "min" else max(raw)
        return _Value(hub, result)

    # Static types --------------------------------------------------------------------------------------------------

    def static_type(self, node):
        """The type of ``node`` without evaluating it: the type of a ``?:`` depends on the branch not taken."""
        if isinstance(node, A.LiteralNode):
            return self._literal(node).type
        if isinstance(node, A.IdentifierNode):
            if node.name in ("true", "false"):
                return BOOL
            slot = self.frame.find(node.name) if self.frame is not None else None
            return slot[0] if slot is not None else self._constant(node.name).type
        if isinstance(node, A.BinaryOpNode):
            operator = str(node.operator)
            if operator in _COMPARISONS or operator in ("&&", "||"):
                return BOOL
            left = _Value(self.static_type(node.left), 0)
            if operator in ("<<", ">>"):
                return promote(_integer_type(left))
            right = _Value(self.static_type(node.right), 0)
            return arithmetic_result(_arithmetic_type(left), _arithmetic_type(right))
        if isinstance(node, A.UnaryOpNode) and not node.is_postfix:
            if str(node.operator) == "!":
                return BOOL
            return promote(_arithmetic_type(_Value(self.static_type(node.operand), 0)))
        if isinstance(node, A.TernaryOpNode):
            true_type = self.static_type(node.true_expr)
            false_type = self.static_type(node.false_expr)
            if true_type == false_type:
                return true_type
            return arithmetic_result(
                _arithmetic_type(_Value(true_type, 0)),
                _arithmetic_type(_Value(false_type, 0)),
            )
        if isinstance(node, A.CastNode):
            return self.scalar_type(node.target_type)
        if isinstance(node, A.FunctionCallNode) and not node.generic_args:
            return self._call_type(node)
        raise _Unresolved(f"the type of {type(node).__name__} is not inferred")

    def _call_type(self, node):
        name = call_name(node)
        arguments = list(node.arguments or [])
        overloads = self.overloads(name)
        if overloads:
            types = [self.static_type(argument) for argument in arguments]
            return self.scalar_type(self._overload(name, overloads, types).return_type)
        if type_from_name(self.aliases.resolve(name)) is not None:
            return self.scalar_type(name)
        if name in _BUILTIN_ARITIES and arguments:
            return self.static_type(arguments[0])
        raise _Unresolved(f"the type of a call of {name} is not inferred")

    # Program functions ---------------------------------------------------------------------------------------------

    def _parameter_type(self, parameter):
        if _qualifiers(parameter) & {"out", "inout", "thread", "device"}:
            raise _Unresolved("parameters passed by reference are not evaluated")
        return self.scalar_type(parameter.param_type)

    def _accepts(self, function, count):
        parameters = list(function.parameters or [])
        if function.body is None or function.generic_params or count > len(parameters):
            return False
        return all(p.default_value is not None for p in parameters[count:])

    def _overload(self, name, overloads, types):
        if is_pure_callee(name):
            raise _Unresolved(f"{name} may also name a built-in function")
        candidates = [f for f in overloads if self._accepts(f, len(types))]
        if len(candidates) == 1:
            return candidates[0]
        exact = []
        for function in candidates:
            try:
                parameter_types = [self._parameter_type(p) for p in function.parameters]
            except _Unresolved:
                continue
            if parameter_types[: len(types)] == types:
                exact.append(function)
        if len(exact) != 1:
            raise _Unresolved(f"no single overload of {name} matches the arguments")
        return exact[0]

    def _invoke(self, name, overloads, arguments):
        values = [self.evaluate(argument) for argument in arguments]
        function = self._overload(name, overloads, [value.type for value in values])
        return_type = self.scalar_type(function.return_type)
        stage_functions = self.stage_functions.get(name) or []
        frame = _Frame(stage_visible=any(function is f for f in stage_functions))
        parameters = list(function.parameters or [])
        for parameter in parameters[len(values) :]:
            values.append(self._default(parameter.default_value))
        for parameter, value in zip(parameters, values):
            hub = self._parameter_type(parameter)
            frame.declare(parameter.name, hub, _convert(value, hub).value)
        saved = (self.frame, self.stage_visible)
        self.frame, self.stage_visible = frame, frame.stage_visible
        try:
            self._enter()
            signal = self._execute(function.body)
        finally:
            self.depth -= 1
            self.frame, self.stage_visible = saved
        if not signal or signal[0] != "return":
            raise _Unresolved(f"{name} does not return a value")
        return _convert(signal[1], return_type)

    def _default(self, expression):
        saved = self.frame
        self.frame = None
        try:
            return self.evaluate(expression)
        finally:
            self.frame = saved

    # Statements ----------------------------------------------------------------------------------------------------

    def _execute(self, node):
        """Runs one statement; returns ``None``, ``("break",)``, ``("continue",)`` or ``("return", value)``."""
        self._step()
        if node is None:
            return None
        if isinstance(node, (A.BlockNode, list)):
            return self._block(
                node.statements if isinstance(node, A.BlockNode) else node
            )
        if isinstance(node, A.VariableNode):
            self._declare(node)
            return None
        if isinstance(node, (A.AssignmentNode, A.ExpressionStatementNode)):
            if getattr(node, "is_tail_expression", False):
                raise _Unresolved("tail expressions are not evaluated")
            self._effect(node)
            return None
        if isinstance(node, A.ReturnNode):
            if node.value is None:
                raise _Unresolved("the function returns no value")
            return ("return", self.evaluate(node.value))
        if isinstance(node, A.IfNode):
            return self._if(node)
        if isinstance(node, (A.ForNode, A.WhileNode, A.DoWhileNode)):
            return self._loop(node)
        if isinstance(node, A.BreakNode):
            return ("break",)
        if isinstance(node, A.ContinueNode):
            return ("continue",)
        if isinstance(node, A.StaticAssertNode):
            # Checked by :mod:`.static_assertions`; it has no run-time effect.
            return None
        raise _Unresolved(f"{type(node).__name__} statements are not evaluated")

    def _block(self, statements):
        self.frame.push()
        try:
            for statement in statements or []:
                signal = self._execute(statement)
                if signal is not None:
                    return signal
            return None
        finally:
            self.frame.pop()

    def _declare(self, node):
        if isinstance(node, A.ArrayNode) or getattr(node, "is_type_alias", False):
            raise _Unresolved("only scalar locals are evaluated")
        if not _qualifiers(node) <= _LOCAL_QUALIFIERS:
            raise _Unresolved("locals with storage qualifiers are not evaluated")
        hub = self.scalar_type(node.var_type)
        value = _UNINITIALIZED
        if node.initial_value is not None:
            if _mentions(node.initial_value, node.name):
                raise _Unresolved(f"the initializer of {node.name} names it")
            value = _convert(self.evaluate(node.initial_value), hub).value
        self.frame.declare(node.name, hub, value)

    def _effect(self, node):
        """Evaluates an expression statement, a ``for`` header item or an assignment, for its side effects."""
        if isinstance(node, A.ExpressionStatementNode):
            node = node.expression
        if isinstance(node, A.AssignmentNode):
            self._assign(node)
        elif isinstance(node, A.UnaryOpNode) and str(node.operator) in ("++", "--"):
            self._increment(node)
        elif isinstance(node, A.VariableNode):
            self._declare(node)
        else:
            self.evaluate(node)

    def _assign(self, node):
        if not isinstance(node.target, A.IdentifierNode):
            raise _Unresolved("only assignments to local scalars are evaluated")
        slot = self._slot(node.target.name)
        operator = str(node.operator or "=")
        value = self.evaluate(node.value)
        if operator in _COMPOUND_OPERATORS:
            if slot[1] is _UNINITIALIZED:
                raise _Unresolved(f"{node.target.name} is read before it is assigned")
            value = _apply(operator[:-1], _Value(slot[0], slot[1]), value)
        elif operator != "=":
            raise _Unresolved(f"assignment operator {operator} is not evaluated")
        slot[1] = _convert(value, slot[0]).value

    def _increment(self, node):
        if not isinstance(node.operand, A.IdentifierNode):
            raise _Unresolved("only local scalars are incremented")
        slot = self._slot(node.operand.name)
        if slot[1] is _UNINITIALIZED or slot[0].kind == "b":
            raise _Unresolved(f"{node.operand.name} cannot be incremented")
        operator = "+" if str(node.operator) == "++" else "-"
        result = _arithmetic(operator, _Value(slot[0], slot[1]), _Value(INT32, 1))
        slot[1] = _convert(result, slot[0]).value

    def _if(self, node):
        if self._truth(node.condition):
            return self._branch(node.then_branch)
        for condition, body in zip(
            node.else_if_conditions or [], node.else_if_bodies or []
        ):
            if self._truth(condition):
                return self._branch(body)
        return self._branch(node.else_branch)

    def _branch(self, body):
        """A substatement is a block scope of its own (C++17 [stmt.stmt]/5)."""
        if isinstance(body, A.BlockNode):
            return self._execute(body)
        return self._block([body])

    def _loop(self, node):
        self.frame.push()
        try:
            if isinstance(node, A.ForNode):
                for item in node.init if isinstance(node.init, list) else [node.init]:
                    if item is not None:
                        self._effect(item)
            first = isinstance(node, A.DoWhileNode)
            while first or node.condition is None or self._truth(node.condition):
                first = False
                signal = self._branch(node.body)
                if signal is not None and signal[0] == "break":
                    break
                if signal is not None and signal[0] == "return":
                    return signal
                update = getattr(node, "update", None)
                for item in update if isinstance(update, list) else [update]:
                    if item is not None:
                        self._effect(item)
                self._step()
            return None
        finally:
            self.frame.pop()


def _literal_node(value):
    """CrossGL spelling of an evaluated value: a literal, or a conversion of one for suffix-less integer types."""
    hub, raw = value
    if hub == BOOL:
        return A.LiteralNode(bool(raw), A.PrimitiveType("bool"))
    if hub == FLOAT32:
        literal = A.LiteralNode(abs(raw), A.PrimitiveType("float"))
        return A.UnaryOpNode("-", literal) if math.copysign(1.0, raw) < 0 else literal
    if hub == UINT32:
        return A.LiteralNode(raw, A.PrimitiveType("uint"))
    if hub == INT32 or -(2**31) <= raw < 2**31:
        literal = A.LiteralNode(abs(raw), A.PrimitiveType("int"))
        if raw == -(2**31):
            literal = A.BinaryOpNode(
                A.UnaryOpNode("-", A.LiteralNode(2**31 - 1, A.PrimitiveType("int"))),
                "-",
                A.LiteralNode(1, A.PrimitiveType("int")),
            )
        elif raw < 0:
            literal = A.UnaryOpNode("-", literal)
    elif 0 <= raw < 2**32:
        literal = A.LiteralNode(raw, A.PrimitiveType("uint"))
    else:
        raise _Unresolved(f"the value {raw} of {hub} has no CrossGL literal")
    if hub == INT32:
        return literal
    return A.FunctionCallNode(A.IdentifierNode(_CONVERSION_SPELLINGS[hub]), [literal])


def _fold(owner, field, key, evaluator):
    node = get_child(owner, field, key)
    if isinstance(node, A.FunctionCallNode) and evaluator.is_user_call(node):
        try:
            replacement = _literal_node(evaluator.fold(node))
        except _Unresolved:
            replacement = None
        if replacement is not None:
            replacement.source_location = getattr(node, "source_location", None)
            set_child(owner, field, key, replacement)
            return 1
    return sum(
        _fold(node, child_field, child_key, evaluator)
        for child_field, child_key in list(child_slots(node))
    )


def _fold_declarations(declarations, evaluator):
    folded = 0
    for node in declarations or []:
        if isinstance(node, A.ConstantNode) and node.value is not None:
            folded += _fold(node, "value", None, evaluator)
        elif _is_compile_time_variable(node):
            folded += _fold(node, "initial_value", None, evaluator)
    return folded


def fold_constant_calls(program, target=None) -> int:
    """Replace calls of program functions in compile-time initializers by their values; returns how many."""
    del target  # Every target receives the values: the calls are valid in none of their constant contexts.
    if not isinstance(program, A.ShaderNode):
        return 0
    aliases = ProgramTypeAliases(program)
    shader_scope = (
        _constant_table(program.global_variables, program.constants),
        _function_table(program.functions),
    )
    evaluator = _Evaluator(aliases, shader_scope)
    folded = _fold_declarations(program.global_variables, evaluator)
    folded += _fold_declarations(program.constants, evaluator)
    for assertion in program.static_assertions or []:
        if (
            isinstance(assertion, A.StaticAssertNode)
            and assertion.condition is not None
        ):
            folded += _fold(assertion, "condition", None, evaluator)
    for stage in program_stages(program):
        stage_scope = (
            _constant_table(getattr(stage, "local_variables", None), ()),
            _function_table(getattr(stage, "local_functions", None)),
        )
        evaluator = _Evaluator(aliases, shader_scope, stage_scope)
        folded += _fold_declarations(getattr(stage, "local_variables", None), evaluator)
    return folded

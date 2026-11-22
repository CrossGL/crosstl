"""Accuracy of the binary64 math helpers (crosstl/translator/semantics/libm_binary64), measured on their CrossGL text.

The helpers compute in binary64 arithmetic, 32-bit integer arithmetic and exact word access. :class:`Evaluator`
evaluates the parsed CrossGL definitions themselves, one numpy lane per operand, with IEEE 754 binary64 operations
(numpy's ``float64``, correctly rounded, never contracted) and CrossGL's ``uint`` arithmetic modulo 2^32. It fails
on any construct it does not model, and on lane states whose meaning a target could choose: a shift by 32 or more,
a signed overflow, a conversion of a negative ``int`` to ``uint`` or of an out-of-range ``double`` to ``int``, and a
floating-point literal that binary32 cannot represent (the GLSL and SPIR-V generators emit unsuffixed literals as
binary32 constants).

Each result is compared with a reference computed in decimal arithmetic at 60 digits (Python's ``decimal``, whose
``exp`` and ``ln`` are correctly rounded), and the error is measured in ulps of the exact value. ``python -m
tests.conformance.libm64_accuracy`` prints the largest error of each helper on the deterministic samples of
:func:`samples`.
"""

from __future__ import annotations

import argparse
import decimal
import functools
import math
import struct

import numpy as np

from crosstl.translator import ast as A
from crosstl.translator.semantics.libm_binary64 import (
    FUNCTIONS,
    helper_program_source,
)

F64 = np.float64
U32 = np.uint32
I32 = np.int32
BOOL = np.bool_

_SCALARS = {"double": F64, "uint": U32, "int": I32, "bool": BOOL}
_VECTORS = {"dvec": F64, "uvec": U32}
_COMPARISONS = {
    "==": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}
_FLOAT = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}
_UNSIGNED = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "&": np.bitwise_and,
    "|": np.bitwise_or,
    "^": np.bitwise_xor,
}
_COMPOUND = {
    f"{operator}=": operator
    for operator in ("+", "-", "*", "/", "&", "|", "^", "<<", ">>")
}


class Unmodelled(Exception):
    """The helper uses a construct, or reaches a lane state, the evaluator does not define."""


def _dtype(value):
    if isinstance(value, tuple):
        return tuple(np.asarray(component).dtype for component in value)
    return np.asarray(value).dtype


def _where(mask, new, old):
    if isinstance(old, tuple):
        if not isinstance(new, tuple) or len(new) != len(old):
            raise Unmodelled("a vector is assigned a value of another shape")
        return tuple(_where(mask, n, o) for n, o in zip(new, old))
    return np.where(mask, new, old)


class _Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.values = {}

    def find(self, name):
        scope = self
        while scope is not None:
            if name in scope.values:
                return scope
            scope = scope.parent
        raise Unmodelled(f"{name} is not declared")


class Evaluator:
    """Evaluates CrossGL functions over numpy lanes: ``double``, ``uint``, ``int``, ``bool``, ``dvecN``, ``uvecN``."""

    def __init__(self, functions):
        self.functions = {function.name: function for function in functions}

    # Types ----------------------------------------------------------------------------------------------------
    @staticmethod
    def dtype_of(type_node):
        if isinstance(type_node, A.VectorType):
            element = getattr(type_node.element_type, "name", None)
            prefix = {"double": "dvec", "uint": "uvec"}.get(element)
            if prefix is None or not str(type_node.size).isdigit():
                raise Unmodelled(f"the type {type_node} is not modelled")
            return (_VECTORS[prefix],) * int(type_node.size)
        name = getattr(type_node, "name", None)
        if name in _SCALARS:
            return _SCALARS[name]
        if name and name[:-1] in _VECTORS and name[-1] in "234":
            return (_VECTORS[name[:-1]],) * int(name[-1])
        raise Unmodelled(f"the type {type_node} is not modelled")

    def typed(self, value, type_node, mask):
        wanted = self.dtype_of(type_node)
        if isinstance(wanted, tuple):
            if not isinstance(value, tuple) or _dtype(value) != tuple(
                np.dtype(w) for w in wanted
            ):
                raise Unmodelled(f"expected {type_node}, got {_dtype(value)}")
            return tuple(
                np.broadcast_to(component, mask.shape).copy() for component in value
            )
        if isinstance(value, tuple) or np.asarray(value).dtype != wanted:
            raise Unmodelled(f"expected {type_node}, got {_dtype(value)}")
        return np.broadcast_to(value, mask.shape).copy()

    @staticmethod
    def need(value, dtype):
        if isinstance(value, tuple) or np.asarray(value).dtype != dtype:
            raise Unmodelled(f"expected {np.dtype(dtype).name}, got {_dtype(value)}")
        return np.asarray(value)

    # Functions ------------------------------------------------------------------------------------------------
    def call(self, name, arguments, mask):
        function = self.functions.get(name)
        if function is None:
            return self.builtin(name, arguments, mask)
        parameters = function.parameters or []
        if len(parameters) != len(arguments):
            raise Unmodelled(f"{name} takes {len(parameters)} arguments")
        scope = _Scope()
        for parameter, argument in zip(parameters, arguments):
            scope.values[parameter.name] = self.typed(
                argument, parameter.param_type, mask
            )
        frame = {"result": None, "returned": np.zeros(mask.shape, bool)}
        self.block(function.body.statements, mask, scope, frame)
        if not np.all(frame["returned"] | ~mask):
            raise Unmodelled(f"{name} can end without a return")
        if frame["result"] is None:
            # No lane is active (a call in the untaken arm of a select): its value is never used.
            wanted = self.dtype_of(function.return_type)
            if isinstance(wanted, tuple):
                return tuple(np.zeros(mask.shape, w) for w in wanted)
            return np.zeros(mask.shape, wanted)
        return self.typed(frame["result"], function.return_type, mask)

    def builtin(self, name, arguments, mask):
        if name == "packDouble2x32":
            (words,) = arguments
            low, high = (self.need(word, U32) for word in words)
            combined = (high.astype(np.uint64) << np.uint64(32)) | low.astype(np.uint64)
            return combined.view(F64)
        if name == "unpackDouble2x32":
            (value,) = arguments
            raw = np.ascontiguousarray(self.need(value, F64)).view(np.uint64)
            return (
                (raw & np.uint64(0xFFFFFFFF)).astype(U32),
                (raw >> np.uint64(32)).astype(U32),
            )
        if name[:-1] in _VECTORS and name[-1] in "234":
            element = _VECTORS[name[:-1]]
            lanes = int(name[-1])
            components = [self.need(a, element) for a in arguments]
            if len(components) == 1:
                components = components * lanes
            if len(components) != lanes:
                raise Unmodelled(f"{name} takes {lanes} components")
            return tuple(np.broadcast_to(c, mask.shape) for c in components)
        if name in ("double", "int", "uint") and len(arguments) == 1:
            return self.convert(name, arguments[0], mask)
        raise Unmodelled(f"the builtin {name} is outside the modelled base")

    def convert(self, name, value, mask):
        array = np.asarray(value)
        active = np.broadcast_to(array, mask.shape)[mask]
        if name == "double":
            if array.dtype not in (I32, U32, F64):
                raise Unmodelled(f"double of {array.dtype}")
            return array.astype(F64)
        if name == "int":
            if array.dtype == U32:
                if np.any(active >= 2**31):
                    raise Unmodelled("int of a uint at or above 2^31")
                return array.astype(I32)
            if array.dtype == F64:
                if not np.all(np.isfinite(active)) or np.any(np.abs(active) >= 2**31):
                    raise Unmodelled("int of a double outside the int range")
                with np.errstate(all="ignore"):
                    return np.where(np.isfinite(array), np.trunc(array), 0).astype(I32)
            raise Unmodelled(f"int of {array.dtype}")
        if array.dtype == I32:
            if np.any(active < 0):
                raise Unmodelled("uint of a negative int")
            return array.astype(U32)
        raise Unmodelled(f"uint of {array.dtype}")

    # Statements -----------------------------------------------------------------------------------------------
    def block(self, statements, mask, scope, frame):
        scope = _Scope(scope)
        for statement in statements:
            active = mask & ~frame["returned"]
            if not active.any():
                return
            self.statement(statement, active, scope, frame)

    def statement(self, node, mask, scope, frame):
        if isinstance(node, A.BlockNode):
            self.block(node.statements, mask, scope, frame)
        elif isinstance(node, A.VariableNode):
            if node.initial_value is None:
                raise Unmodelled(f"{node.name} has no initial value")
            value = self.eval(node.initial_value, mask, scope)
            scope.values[node.name] = self.typed(value, node.var_type, mask)
        elif isinstance(node, A.AssignmentNode):
            self.assign(node, mask, scope)
        elif isinstance(node, A.ExpressionStatementNode) and isinstance(
            node.expression, A.AssignmentNode
        ):
            self.assign(node.expression, mask, scope)
        elif isinstance(node, A.IfNode):
            condition = self.need(self.eval(node.condition, mask, scope), BOOL)
            self.statement(node.then_branch, mask & condition, scope, frame)
            if node.else_branch is not None:
                self.statement(node.else_branch, mask & ~condition, scope, frame)
        elif isinstance(node, A.ReturnNode):
            value = self.eval(node.value, mask, scope)
            if isinstance(value, tuple):
                value = tuple(np.broadcast_to(c, mask.shape) for c in value)
            else:
                value = np.broadcast_to(value, mask.shape)
            if frame["result"] is None:
                frame["result"] = (
                    tuple(np.zeros(mask.shape, c.dtype) for c in value)
                    if isinstance(value, tuple)
                    else np.zeros(mask.shape, value.dtype)
                )
            elif _dtype(frame["result"]) != _dtype(value):
                raise Unmodelled("the returns have different types")
            frame["result"] = _where(mask, value, frame["result"])
            frame["returned"] = frame["returned"] | mask
        else:
            raise Unmodelled(f"the statement {type(node).__name__} is not modelled")

    def assign(self, node, mask, scope):
        if not isinstance(node.target, A.IdentifierNode):
            raise Unmodelled("only assignments to a local are modelled")
        owner = scope.find(node.target.name)
        old = owner.values[node.target.name]
        value = self.eval(node.value, mask, scope)
        if node.operator != "=":
            operator = _COMPOUND.get(node.operator)
            if operator is None:
                raise Unmodelled(f"the assignment {node.operator} is not modelled")
            value = self.binary(operator, old, value, mask)
        if _dtype(value) != _dtype(old):
            raise Unmodelled(f"{node.target.name} is assigned another type")
        owner.values[node.target.name] = _where(mask, value, old)

    # Expressions ----------------------------------------------------------------------------------------------
    def eval(self, node, mask, scope):
        if isinstance(node, A.LiteralNode):
            return self.literal(node)
        if isinstance(node, A.IdentifierNode):
            return scope.find(node.name).values[node.name]
        if isinstance(node, A.BinaryOpNode):
            if node.operator in ("&&", "||"):
                left = self.need(self.eval(node.left, mask, scope), BOOL)
                taken = mask & (left if node.operator == "&&" else ~left)
                right = self.need(self.eval(node.right, taken, scope), BOOL)
                return left & right if node.operator == "&&" else left | right
            left = self.eval(node.left, mask, scope)
            right = self.eval(node.right, mask, scope)
            return self.binary(node.operator, left, right, mask)
        if isinstance(node, A.UnaryOpNode):
            operand = self.eval(node.operand, mask, scope)
            if node.operator == "~":
                return ~self.need(operand, U32)
            if node.operator == "!":
                return ~self.need(operand, BOOL)
            if node.operator == "-":
                array = np.asarray(operand)
                if array.dtype == F64:
                    return np.negative(array)
                if array.dtype == I32:
                    if np.any(np.broadcast_to(array, mask.shape)[mask] == -(2**31)):
                        raise Unmodelled("a negation overflows int")
                    return np.negative(array)
            raise Unmodelled(f"the unary operator {node.operator} is not modelled")
        if isinstance(node, A.TernaryOpNode):
            condition = self.need(self.eval(node.condition, mask, scope), BOOL)
            true = self.eval(node.true_expr, mask & condition, scope)
            false = self.eval(node.false_expr, mask & ~condition, scope)
            if _dtype(true) != _dtype(false):
                raise Unmodelled("the arms of a select have different types")
            return np.where(condition, true, false)
        if isinstance(node, A.MemberAccessNode):
            value = self.eval(node.object_expr, mask, scope)
            if isinstance(value, tuple) and node.member in "xyzw"[: len(value)]:
                return value["xyzw".index(node.member)]
            raise Unmodelled(f"the member {node.member} is not modelled")
        if isinstance(node, A.FunctionCallNode):
            name = node.function if isinstance(node.function, str) else None
            name = name or getattr(node.function, "name", None)
            arguments = [
                self.eval(argument, mask, scope) for argument in node.arguments
            ]
            return self.call(name, arguments, mask)
        raise Unmodelled(f"the expression {type(node).__name__} is not modelled")

    @staticmethod
    def literal(node):
        name = getattr(node.literal_type, "name", None)
        value = node.value
        if name == "uint" and isinstance(value, int) and 0 <= value < 2**32:
            return U32(value)
        if name == "int" and isinstance(value, int) and -(2**31) <= value < 2**31:
            return I32(value)
        if name == "bool" and isinstance(value, bool):
            return BOOL(value)
        if name == "float" and isinstance(value, float):
            if float(np.float32(value)) != value:
                raise Unmodelled(f"the literal {value!r} is not exact in binary32")
            return F64(value)
        raise Unmodelled(f"the literal {value!r} of type {name} is not modelled")

    def binary(self, operator, left, right, mask):
        left, right = np.asarray(left), np.asarray(right)
        if left.dtype != right.dtype:
            raise Unmodelled(f"{operator} on {left.dtype} and {right.dtype}")
        if operator in _COMPARISONS:
            return _COMPARISONS[operator](left, right)
        if left.dtype == F64:
            function = _FLOAT.get(operator)
            if function is None:
                raise Unmodelled(f"the operator {operator} on double is not modelled")
            with np.errstate(all="ignore"):
                return function(left, right)
        if left.dtype == I32:
            if operator not in ("+", "-"):
                raise Unmodelled(f"the operator {operator} on int is not modelled")
            wide = (
                left.astype(np.int64) + right.astype(np.int64)
                if operator == "+"
                else left.astype(np.int64) - right.astype(np.int64)
            )
            active = np.broadcast_to(wide, mask.shape)[mask]
            if np.any((active < -(2**31)) | (active >= 2**31)):
                raise Unmodelled(f"a signed {operator} overflows")
            return wide.astype(I32)
        if left.dtype != U32:
            raise Unmodelled(f"the operator {operator} on {left.dtype} is not modelled")
        if operator in ("<<", ">>"):
            if np.any(np.broadcast_to(right, mask.shape)[mask] >= 32):
                raise Unmodelled(f"a lane shifts by 32 or more with {operator}")
            amount = right & U32(31)
            return left << amount if operator == "<<" else left >> amount
        function = _UNSIGNED.get(operator)
        if function is None:
            raise Unmodelled(f"the operator {operator} on uint is not modelled")
        return function(left, right, dtype=U32)


@functools.lru_cache(maxsize=None)
def evaluator():
    """An evaluator holding every helper of the package."""
    from crosstl.translator.source_registry import (
        SOURCE_REGISTRY,
        register_default_sources,
    )

    register_default_sources()
    names = [helper for helper, _ in FUNCTIONS.values()]
    program = SOURCE_REGISTRY.get("cgl").parse(helper_program_source(names))
    return Evaluator(program.functions)


def evaluate(function, *operands):
    """The helper of the CrossGL built-in ``function`` on the binary64 ``operands`` (equal-length sequences)."""
    helper, arity = FUNCTIONS[function]
    arrays = [np.ascontiguousarray(np.asarray(operand, F64)) for operand in operands]
    if len(arrays) != arity:
        raise ValueError(f"{function} takes {arity} operands")
    mask = np.ones(arrays[0].shape, bool)
    with np.errstate(all="ignore"):
        return evaluator().call(helper, arrays, mask)


# References -----------------------------------------------------------------------------------------------------
_CONTEXT = decimal.Context(
    prec=60,
    Emax=decimal.MAX_EMAX,
    Emin=decimal.MIN_EMIN,
    traps=[decimal.InvalidOperation, decimal.DivisionByZero],
)
_LN2 = decimal.Decimal(2).ln(_CONTEXT)


def reference(function, *operands):
    """The exact value (a ``Decimal``) of ``function`` at finite operands where it is finite and nonzero."""
    values = [decimal.Decimal(operand) for operand in operands]
    context = _CONTEXT
    if function == "exp":
        return values[0].exp(context)
    if function == "exp2":
        return context.multiply(values[0], _LN2).exp(context)
    if function == "log":
        return values[0].ln(context)
    if function == "log2":
        return context.divide(values[0].ln(context), _LN2)
    if function == "pow":
        x, y = values
        magnitude = context.multiply(y, x.copy_abs().ln(context)).exp(context)
        if x < 0:
            if y != y.to_integral_value():
                raise ValueError("a negative base needs an integral exponent")
            if int(y) % 2:
                return magnitude.copy_negate()
        return magnitude
    raise KeyError(function)


_MAX = decimal.Decimal(np.finfo(F64).max.item())
_TWO = decimal.Decimal(2)


def ulp_error(result, exact):
    """The error of the binary64 ``result`` in ulps of the exact value ``exact`` (a ``Decimal``)."""
    result = float(result)
    if math.isnan(result):
        return math.inf
    magnitude = exact.copy_abs()
    if magnitude > _MAX:
        # Beyond the largest finite value, an ulp is that of the top binade, and infinity stands for 2^1024.
        if math.isinf(result) and (result > 0) == (exact > 0):
            return (
                float(abs(magnitude - _TWO**1024) / _TWO**971)
                if magnitude < _TWO**1024
                else 0.0
            )
        return float(abs(decimal.Decimal(result) - exact) / _TWO**971)
    if math.isinf(result):
        return float(abs(_TWO**1024 - magnitude) / _TWO**971)
    rounded = float(magnitude)
    exponent = math.frexp(rounded)[1] - 1 if rounded else -1074
    unit = _TWO ** (max(exponent, -1022) - 52)
    return float(abs(decimal.Decimal(result) - exact) / unit)


def max_error(function, *operands):
    """The largest ulp error of the helper over the operands, and the operands where it occurs."""
    results = evaluate(function, *operands)
    worst, where = 0.0, None
    for index, result in enumerate(results):
        point = tuple(float(operand[index]) for operand in operands)
        error = ulp_error(result, reference(function, *point))
        if error > worst:
            worst, where = error, point
    return worst, where


# Samples --------------------------------------------------------------------------------------------------------
def _from_words(high, low):
    return np.array(
        [
            struct.unpack("<d", struct.pack("<II", int(l), int(h)))[0]
            for h, l in zip(high, low)
        ]
    )


def _neighbours(values, count=3):
    """Each value and its ``count`` nearest binary64 neighbours on each side."""
    values = np.asarray(values, F64)
    result = [values]
    up, down = values.copy(), values.copy()
    for _ in range(count):
        up = np.nextafter(up, np.inf)
        down = np.nextafter(down, -np.inf)
        result += [up, down]
    return np.concatenate(result)


def _positive_patterns(rng, count):
    """Positive finite binary64 values with uniformly random exponents (subnormals included) and significands."""
    high = rng.integers(0, 0x7FF00000, count)
    low = rng.integers(0, 2**32, count)
    return _from_words(high, low)


def samples(function, count=1500, seed=0):
    """Deterministic finite operands of ``function`` whose result is finite and nonzero, covering its branches."""
    rng = np.random.default_rng(seed)
    if function == "exp":
        ln2 = math.log(2)
        boundaries = [
            709.782712893384,
            -745.1332191019411,
            -708.3964185322641,
            0.5 * ln2,
            1.5 * ln2,
            2.0**-28,
            -(2.0**-28),
            -0.5 * ln2,
            -1.5 * ln2,
        ]
        multiples = np.arange(-1074, 1024) * ln2
        values = np.concatenate(
            [
                rng.uniform(-745.13, 709.78, count),
                rng.uniform(-1.1, 1.1, count // 3),
                _neighbours(boundaries),
                _neighbours(multiples[::7], 1),
                np.ldexp(rng.uniform(-1, 1, 200), rng.integers(-1074, -20, 200)),
            ]
        )
        values = values[(values > -745.13) & (values < 709.78)]
        return (values,)
    if function == "exp2":
        values = np.concatenate(
            [
                rng.uniform(-1074.9, 1023.99, count),
                rng.uniform(-1.1, 1.1, count // 3),
                _neighbours([1023.0, -1022.0, -1074.0, -1074.5, 0.5, -0.5, 1.0, -1.0]),
                _neighbours(np.arange(-1070, 1024, 9.5), 1),
                np.ldexp(rng.uniform(-1, 1, 200), rng.integers(-1074, -20, 200)),
            ]
        )
        values = values[(values > -1075.0) & (values < 1024.0)]
        return (values,)
    if function in ("log", "log2"):
        near_one = 1.0 + np.ldexp(
            rng.uniform(-1, 1, count // 2), rng.integers(-52, -1, count // 2)
        )
        values = np.concatenate(
            [
                _positive_patterns(rng, count),
                near_one,
                _neighbours(
                    [
                        1.0,
                        math.sqrt(2),
                        math.sqrt(0.5),
                        1.5,
                        2.0,
                        0.5,
                        2.0**-1022,
                        2.0**-1074,
                    ]
                ),
                np.ldexp(1.0, np.arange(-1074, 1024, 5)),
                np.ldexp(rng.uniform(0, 1, 200), -1022),
            ]
        )
        values = values[(values > 0) & np.isfinite(values) & (values != 1.0)]
        return (values,)
    if function == "pow":
        bases = np.concatenate(
            [_positive_patterns(rng, count), rng.uniform(0.25, 4.0, count // 2)]
        )
        bases = bases[bases != 1.0]
        targets = rng.uniform(-1074.0, 1023.0, bases.size)
        exponents = targets / np.log2(bases)
        # Bases within 2^-20 of one with exponents beyond 2^31 (e_pow.c's series branch) and below it.
        near = 1.0 + np.ldexp(rng.uniform(-1, 1, 600), rng.integers(-53, -18, 600))
        with np.errstate(divide="ignore"):
            near_exponents = rng.uniform(-1070, 1020, 600) / np.log2(near)
        negative = -rng.uniform(0.5, 3.0, 300)
        integers = rng.integers(-60, 60, 300).astype(F64)
        small = rng.uniform(0.1, 10.0, 300)
        small_exponents = rng.uniform(-30, 30, 300)
        x = np.concatenate([bases, near, negative, small, [2.0, 10.0, 0.5, 3.0]])
        y = np.concatenate(
            [
                exponents,
                near_exponents,
                integers,
                small_exponents,
                [0.5, -1.0, 3.0, 1.0 / 3.0],
            ]
        )
        keep = np.isfinite(y) & (y != 0)
        x, y = x[keep], y[keep]
        with np.errstate(all="ignore"):
            magnitude = np.abs(y * np.log2(np.abs(x)))
        keep = (
            (y * np.log2(np.abs(x)) > -1074.0)
            & (y * np.log2(np.abs(x)) < 1023.9)
            & (magnitude > 0)
        )
        return (x[keep], y[keep])
    raise KeyError(function)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    for function in FUNCTIONS:
        operands = samples(function, arguments.count, arguments.seed)
        worst, where = max_error(function, *operands)
        print(
            f"{function}: {operands[0].size} operands, max {worst:.4f} ulp at {where}"
        )


if __name__ == "__main__":
    main()

"""Exhaustive checks of the libm rule helpers over every binary32 operand.

The libm family's polyfills (crosstl/translator/semantics/libm_trig.py) compute in unsigned 32-bit integer arithmetic,
which every target defines exactly. An SMT query over the ulp error of a transcendental function is out of reach, but
the operand domain is finite: this module evaluates the parsed CrossGL helper itself, one numpy lane per operand,
for all 2^32 bit patterns, and measures each result against the binary64 reference.

The evaluator follows CrossGL's integer semantics (``uint`` arithmetic modulo 2^32) and fails on any construct it
does not model, on a shift amount of 32 or more in a lane that executes it, and on any floating-point arithmetic:
the helpers' trusted base is integer arithmetic, comparison, select and bit reinterpretation.

``python -m tests.conformance.libm_exhaustive`` proves every helper and rewrites ``libm_proofs.json``; the rule
table test checks that each certificate names the helper text it was computed for, and re-evaluates a stratified
sample on every run.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from crosstl.translator import ast as A
from crosstl.translator.semantics import rule_table
from crosstl.translator.semantics.helpers import helper_program_source

PROOFS = Path(__file__).with_name("libm_proofs.json")
U32 = np.uint32
F32 = np.float32
_COMPARISONS = {
    "==": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}
_INTEGER = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "&": np.bitwise_and,
    "|": np.bitwise_or,
    "^": np.bitwise_xor,
}
_COMPOUND = {f"{operator}=": operator for operator in (*_INTEGER, "<<", ">>")}
# The reference functions, by the CrossGL builtin a native entry calls.
REFERENCE = {"sin": np.sin, "cos": np.cos}


class Unmodelled(Exception):
    """The helper uses a construct, or reaches a lane state, the evaluator does not define."""


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
    """Evaluates CrossGL functions over numpy lanes: ``uint`` as uint32, ``bool`` as bool, ``float`` as float32."""

    def __init__(self, functions):
        self.functions = {function.name: function for function in functions}

    # Functions ------------------------------------------------------------------------------------------------
    def call(self, name, arguments, mask):
        function = self.functions.get(name)
        if function is None:
            return self.builtin(name, arguments)
        parameters = function.parameters or []
        if len(parameters) != len(arguments):
            raise Unmodelled(f"{name} takes {len(parameters)} arguments")
        scope = _Scope()
        for parameter, argument in zip(parameters, arguments):
            scope.values[parameter.name] = self.typed(argument, parameter.param_type)
        frame = {"result": None, "returned": np.zeros(mask.shape, bool)}
        self.block(function.body.statements, mask, scope, frame)
        if not np.all(frame["returned"] | ~mask):
            raise Unmodelled(f"{name} can end without a return")
        return frame["result"]

    def builtin(self, name, arguments):
        (value,) = arguments
        if name == "asuint":
            return self.need(value, F32).view(U32)
        if name == "asfloat":
            return self.need(value, U32).view(F32)
        raise Unmodelled(f"the builtin {name} is outside the integer base")

    @staticmethod
    def typed(value, type_node):
        name = getattr(type_node, "name", None)
        dtype = {"uint": U32, "bool": np.bool_, "float": F32}.get(name)
        if dtype is None:
            raise Unmodelled(f"the type {type_node} is not modelled")
        return Evaluator.need(value, dtype)

    @staticmethod
    def need(value, dtype):
        array = np.asarray(value)
        if array.dtype != dtype:
            raise Unmodelled(f"expected {np.dtype(dtype).name}, got {array.dtype.name}")
        return array

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
            value = self.typed(
                self.eval(node.initial_value, mask, scope), node.var_type
            )
            scope.values[node.name] = np.broadcast_to(value, mask.shape).copy()
        elif isinstance(node, A.AssignmentNode):
            self.assign(node, mask, scope)
        elif isinstance(node, A.ExpressionStatementNode) and isinstance(
            node.expression, A.AssignmentNode
        ):
            self.assign(node.expression, mask, scope)
        elif isinstance(node, A.IfNode):
            condition = self.need(self.eval(node.condition, mask, scope), np.bool_)
            self.statement(node.then_branch, mask & condition, scope, frame)
            if node.else_branch is not None:
                self.statement(node.else_branch, mask & ~condition, scope, frame)
        elif isinstance(node, A.ReturnNode):
            value = np.broadcast_to(self.eval(node.value, mask, scope), mask.shape)
            if frame["result"] is None:
                frame["result"] = np.zeros(mask.shape, value.dtype)
            elif frame["result"].dtype != value.dtype:
                raise Unmodelled("the returns have different types")
            frame["result"] = np.where(mask, value, frame["result"])
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
        value = self.need(np.broadcast_to(value, mask.shape), old.dtype)
        owner.values[node.target.name] = np.where(mask, value, old)

    # Expressions ----------------------------------------------------------------------------------------------
    def eval(self, node, mask, scope):
        if isinstance(node, A.LiteralNode):
            return self.literal(node)
        if isinstance(node, A.IdentifierNode):
            return scope.find(node.name).values[node.name]
        if isinstance(node, A.BinaryOpNode):
            if node.operator in ("&&", "||"):
                left = self.need(self.eval(node.left, mask, scope), np.bool_)
                # Short-circuit: the right operand runs only where it decides the result.
                taken = mask & (left if node.operator == "&&" else ~left)
                right = self.need(self.eval(node.right, taken, scope), np.bool_)
                return left & right if node.operator == "&&" else left | right
            left = self.eval(node.left, mask, scope)
            right = self.eval(node.right, mask, scope)
            return self.binary(node.operator, left, right, mask)
        if isinstance(node, A.UnaryOpNode):
            operand = self.eval(node.operand, mask, scope)
            if node.operator == "~":
                return ~self.need(operand, U32)
            if node.operator == "!":
                return ~self.need(operand, np.bool_)
            raise Unmodelled(f"the unary operator {node.operator} is not modelled")
        if isinstance(node, A.TernaryOpNode):
            condition = self.need(self.eval(node.condition, mask, scope), np.bool_)
            true = self.eval(node.true_expr, mask & condition, scope)
            false = self.eval(node.false_expr, mask & ~condition, scope)
            if np.asarray(true).dtype != np.asarray(false).dtype:
                raise Unmodelled("the arms of a select have different types")
            return np.where(condition, true, false)
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
        if name == "uint" and isinstance(node.value, int) and 0 <= node.value < 2**32:
            return U32(node.value)
        if name == "bool" and isinstance(node.value, bool):
            return np.bool_(node.value)
        if name == "float" and isinstance(node.value, float):
            return F32(node.value)
        raise Unmodelled(f"the literal {node.value!r} of type {name} is not modelled")

    def binary(self, operator, left, right, mask):
        if operator in _COMPARISONS:
            left, right = np.asarray(left), np.asarray(right)
            if left.dtype != right.dtype or left.dtype == F32:
                raise Unmodelled(
                    f"the comparison {operator} needs two integer or bool operands"
                )
            return _COMPARISONS[operator](left, right)
        left, right = self.need(left, U32), self.need(right, U32)
        if operator in ("<<", ">>"):
            if np.any(np.broadcast_to(right, mask.shape)[mask] >= 32):
                raise Unmodelled(f"a lane shifts by 32 or more with {operator}")
            amount = right & U32(31)
            return left << amount if operator == "<<" else left >> amount
        function = _INTEGER.get(operator)
        if function is None:
            raise Unmodelled(f"the operator {operator} is not modelled")
        return function(left, right, dtype=U32)


def ulp_error(result, operand, reference):
    """The error of each binary32 ``result`` in ulps of the exact value, approximated by the binary64 reference."""
    exact = reference(operand.astype(np.float64))
    exponent = np.floor(np.log2(np.maximum(np.abs(exact), 2.0**-126)))
    return np.abs(result.astype(np.float64) - exact) / np.exp2(exponent - 23)


@functools.lru_cache(maxsize=None)
def evaluator_for(intrinsic):
    from crosstl.translator.source_registry import (
        SOURCE_REGISTRY,
        register_default_sources,
    )

    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(helper_program_source(intrinsic))
    return Evaluator(program.functions), rule_table.HELPERS[intrinsic].name


def check(intrinsic, bits):
    """Evaluate the helper of ``intrinsic`` on the binary32 patterns ``bits``; returns (max ulp, its pattern)."""
    evaluator, name = evaluator_for(intrinsic)
    operator, _ = rule_table.INTRINSICS[intrinsic]
    operand = bits.astype(U32).view(F32)
    mask = np.ones(operand.shape, bool)
    result = evaluator.call(name, [operand], mask)
    finite = np.isfinite(operand)
    if not np.all(np.isnan(result[~finite])):
        raise AssertionError(
            f"{name} keeps a non-NaN result for an infinite or NaN operand"
        )
    if operator == "sin":
        zero = finite & (operand == 0)
        if not np.array_equal(result[zero].view(U32), operand[zero].view(U32)):
            raise AssertionError(f"{name} loses the sign of a zero operand")
    error = np.zeros(operand.shape)
    error[finite] = ulp_error(result[finite], operand[finite], REFERENCE[operator])
    worst = int(np.argmax(error))
    return float(error[worst]), int(bits[worst])


def sample_bits(count_per_exponent=64, seed=0):
    """Every exponent of both signs with random significands, the significand extremes, and the special values."""
    rng = np.random.default_rng(seed)
    exponents = np.arange(256, dtype=np.uint64) << np.uint64(23)
    mantissas = np.concatenate(
        [
            np.array([0, 1, 0x7FFFFF], np.uint64),
            rng.integers(0, 1 << 23, count_per_exponent, dtype=np.uint64),
        ]
    )
    bits = (exponents[:, None] | mantissas[None, :]).ravel()
    return np.concatenate([bits, bits | np.uint64(0x80000000)]).astype(U32)


def helper_sha256(intrinsic):
    return hashlib.sha256(helper_program_source(intrinsic).encode()).hexdigest()


def _check_chunk(job):
    intrinsic, first, count = job
    bits = np.arange(first, first + count, dtype=np.uint64).astype(U32)
    return check(intrinsic, bits)


def prove(intrinsic, chunk_bits=22, jobs=1, log=None):
    """The certificate of the helper of ``intrinsic`` over all 2^32 operands."""
    from concurrent.futures import ProcessPoolExecutor

    start = time.monotonic()
    worst = (0.0, 0)
    step = 1 << chunk_bits
    work = [(intrinsic, first, step) for first in range(0, 1 << 32, step)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for done, result in enumerate(pool.map(_check_chunk, work), 1):
            if result[0] > worst[0]:
                worst = result
            if log is not None and done % 64 == 0:
                print(
                    f"{intrinsic} {done}/{len(work)} worst {worst[0]:.4f} ulp",
                    file=log,
                    flush=True,
                )
    return {
        "helperSHA256": helper_sha256(intrinsic),
        "operands": 1 << 32,
        "maxUlp": round(worst[0], 4),
        "maxUlpOperand": f"{worst[1]:#010x}",
        "reference": "numpy binary64 " + rule_table.INTRINSICS[intrinsic][0],
        "seconds": round(time.monotonic() - start),
    }


def load_proofs():
    return json.loads(PROOFS.read_text(encoding="utf-8")) if PROOFS.exists() else {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "intrinsics", nargs="*", default=sorted(rule_table.LIBM_INTRINSICS)
    )
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes")
    args = parser.parse_args(argv)
    for intrinsic in args.intrinsics:
        proof = prove(intrinsic, jobs=args.jobs, log=sys.stderr)
        print(intrinsic, json.dumps(proof), flush=True)
        # Each certificate is written as soon as it exists, so an interrupted run keeps the finished ones.
        proofs = load_proofs()
        proofs[intrinsic] = proof
        PROOFS.write_text(
            json.dumps(proofs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

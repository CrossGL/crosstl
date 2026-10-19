"""SMT check of the rule table (crosstl/translator/semantics/rule_table.py) with Z3.

For every native and polyfill entry, ask Z3 for operands on which the target construct does not refine the carried
rule, in the style of Alive2:

    find x :  not ( defined_T(x) and result_T(x) == rule_S(x) )

``rule_S`` is the carried rule. For ``wrap`` it is the low-order 32 bits of the exact result; for ``udiv_zero`` it is
the unsigned quotient or remainder, and 0xFFFFFFFF for a zero divisor (Direct3D 11.3 section 22.12.9). Both are
defined for every input. The target construct is not modelled by hand: a native entry is the hub operation
``x0 op x1``, and a polyfill entry is the template the lowering pass instantiates or the body of the helper function
it injects, parsed by the CrossGL parser. Each is evaluated symbolically under the target's semantics of its trusted
base (below). Satisfiable means unsound and comes with a counterexample; unsatisfiable means sound under the model.

Target model:
  - unsigned 32-bit ``+``, ``-``, ``*`` are modular on every target (MSL 4.1 section 3.1, C++17 [basic.fundamental]
    for CUDA, HIP and Slang's C-like targets, GLSL 4.60 section 4.1.3, SPIR-V OpIAdd, WGSL section 6.2.3, Mojo
    UInt32 by reading);
  - signed ``+``, ``-``, ``*`` and unary ``-`` wrap on GLSL, GLSL ES and WGSL; they are undefined on overflow on
    Metal, DirectX (DXC marks them nsw), CUDA, HIP, Slang and Mojo, and on Vulkan, where the SPIR-V rule wraps but
    MoltenVK and KosmicKrisp hand the operation to MSL; on Rust they wrap under ``overflow-checks = false``, the
    configuration the entry records, and panic otherwise;
  - ``asint`` and ``asuint`` reinterpret the 32 bits exactly, and integer literals are exact;
  - unsigned ``/`` and ``%`` are exact for a nonzero divisor. A zero divisor gives an unspecified value on GLSL and
    GLSL ES (GLSL 4.60 and ESSL 3.00 section 5.9), the dividend for ``/`` and 0 for ``%`` on WGSL (section 8.7), a
    panic on Rust, and undefined behaviour everywhere else (MSL and CUDA/HIP C++17 [expr.mul]/4, SPIR-V OpUDiv and
    OpUMod, DXC's LLVM udiv and urem, Slang, Mojo by reading);
  - ``==`` and ``!=`` compare bits, and a conditional is evaluated eagerly: a target may compute both arms (SPIR-V
    OpSelect, WGSL select), so an undefined operation in either arm makes the whole construct undefined;
  - ``&``, ``<``, ``<=``, ``>`` and ``>=`` act on 32-bit integers, signed only when both operands are signed;
  - a binary32 value is its bit pattern: ``asfloat``, and ``asuint`` or ``asint`` of a float, reinterpret it exactly,
    and ``int(f)`` and ``uint(f)`` truncate toward zero and are defined only where the truncated value is
    representable (C++17 [conv.fpint]/1 for MSL, CUDA and HIP; DXC's fptosi and fptoui give poison otherwise; GLSL,
    SPIR-V and WGSL are taken to leave it undefined too). Floating-point comparison and arithmetic are not modelled.

For ``convert_sat`` the carried rule is OpenCL C 6.4.3.3, stated over IEEE binary32 values: NaN gives 0, and other
values truncate toward zero and clamp to the result range.

The check also fails when an entry is missing for a registered backend, when a template uses a parameter other than
exactly once (helper functions may repeat parameters, since a call evaluates each operand once), when an intrinsic
has neither a template nor a helper, or when a construct of a template has no model here. Run ``python -m tests.conformance.rule_table_smt``;
it exits with status 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import z3

from crosstl.translator import ast as A
from crosstl.translator.semantics import rule_table
from crosstl.translator.semantics.helpers import helper_body
from crosstl.translator.semantics.hub_types import call_name, type_from_node
from crosstl.translator.semantics.lower import template

BITS = 32
ALL_ONES = (1 << BITS) - 1
SIGNED_WRAPS = frozenset({"opengl", "webgl", "wgsl"})
# A configuration under which a target's signed arithmetic wraps; the entry must record it.
SIGNED_WRAPS_UNDER = {"rust": "overflow-checks = false"}
# What an unsigned division or remainder by zero gives; every target not listed leaves it undefined (or panics).
UNSIGNED_DIVISION_BY_ZERO = {
    "opengl": "unspecified",
    "webgl": "unspecified",
    "wgsl": "wgsl",
}


class Unmodelled(Exception):
    """A template construct the checker has no semantics for."""


@dataclass(frozen=True)
class Value:
    term: object
    kind: str  # "i", "u" or "b"


class Semantics:
    """Symbolic evaluation of a hub expression under one target's semantics; ``defined`` collects UB conditions."""

    def __init__(self, target, configuration, bindings):
        self.target = target
        self.bindings = bindings
        self.defined = []
        self.signed_wraps = target in SIGNED_WRAPS or (
            configuration is not None
            and SIGNED_WRAPS_UNDER.get(target) == configuration
        )

    def eval(self, node) -> Value:
        if isinstance(node, A.IdentifierNode):
            if node.name not in self.bindings:
                raise Unmodelled(f"free identifier {node.name!r}")
            return self.bindings[node.name]
        if isinstance(node, A.LiteralNode):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, int):
                raise Unmodelled(f"literal {value!r}")
            declared = type_from_node(node.literal_type)
            kind = declared.kind if declared is not None else "i"
            if kind not in ("i", "u") or (declared and declared.bits != BITS):
                raise Unmodelled(f"literal type {node.literal_type!r}")
            return Value(z3.BitVecVal(value % (1 << BITS), BITS), kind)
        if isinstance(node, A.FunctionCallNode):
            name = call_name(node)
            if (
                name not in ("asint", "asuint", "asfloat", "int", "uint")
                or len(node.arguments) != 1
            ):
                raise Unmodelled(f"call {name!r}")
            operand = self.eval(node.arguments[0])
            if operand.kind == "b":
                raise Unmodelled(f"boolean operand of {name}")
            if name == "asfloat":
                if operand.kind == "f":
                    raise Unmodelled("asfloat of a float")
                return Value(operand.term, "f")
            if operand.kind == "f" and name in ("int", "uint"):
                return self.float_to_integer(name, operand)
            return Value(operand.term, "i" if name in ("asint", "int") else "u")
        if isinstance(node, A.BinaryOpNode) and node.op in ("+", "-", "*"):
            return self.arithmetic(node.op, self.eval(node.left), self.eval(node.right))
        if isinstance(node, A.BinaryOpNode) and node.op in ("/", "%"):
            return self.division(node.op, self.eval(node.left), self.eval(node.right))
        if isinstance(node, A.BinaryOpNode) and node.op == "&":
            left, right = self.integers(node.op, node.left, node.right)
            kind = "u" if "u" in (left.kind, right.kind) else "i"
            return Value(left.term & right.term, kind)
        if isinstance(node, A.BinaryOpNode) and node.op in _ORDERINGS:
            left, right = self.integers(node.op, node.left, node.right)
            signed = left.kind == right.kind == "i"
            return Value(_ORDERINGS[node.op](left.term, right.term, signed), "b")
        if isinstance(node, A.BinaryOpNode) and node.op in ("==", "!="):
            left, right = self.integers(node.op, node.left, node.right)
            equal = left.term == right.term
            return Value(equal if node.op == "==" else z3.Not(equal), "b")
        if isinstance(node, A.TernaryOpNode):
            condition = self.eval(node.condition)
            if condition.kind != "b":
                raise Unmodelled("non-boolean condition")
            chosen, other = self.eval(node.true_expr), self.eval(node.false_expr)
            if chosen.kind != other.kind or chosen.kind == "b":
                raise Unmodelled(f"conditional of {chosen.kind} and {other.kind}")
            return Value(z3.If(condition.term, chosen.term, other.term), chosen.kind)
        if isinstance(node, A.UnaryOpNode) and node.op == "-":
            operand = self.eval(node.operand)
            if operand.kind not in ("i", "u"):
                raise Unmodelled(f"negation of a {operand.kind!r} value")
            if operand.kind == "i" and not self.signed_wraps:
                self.defined.append(z3.BVSNegNoOverflow(operand.term))
            return Value(-operand.term, operand.kind)
        raise Unmodelled(f"{type(node).__name__} {getattr(node, 'op', '')}".strip())

    def integers(self, op, left_node, right_node):
        """Both operands of an integer-only operator; ``==`` on floats is not bit equality, so it is unmodelled."""
        left, right = self.eval(left_node), self.eval(right_node)
        if not {left.kind, right.kind} <= {"i", "u"}:
            raise Unmodelled(f"{left.kind!r} and {right.kind!r} operands of {op}")
        return left, right

    def float_to_integer(self, name, operand) -> Value:
        """``int(f)`` or ``uint(f)``: truncation toward zero, defined only when the truncated value is representable.

        NaN and out-of-range operands are undefined behaviour in C++17 [conv.fpint]/1 (MSL, CUDA, HIP), poison for
        DXC's LLVM fptosi and fptoui, and unspecified on GLSL, SPIR-V and WGSL; the model treats them all as undefined.
        """
        value = z3.fpBVToFP(operand.term, z3.Float32())
        if name == "int":
            low, high = _fp(-(2.0**31)), _fp(2.0**31)
            self.defined.append(z3.And(z3.fpGEQ(value, low), z3.fpLT(value, high)))
            return Value(z3.fpToSBV(z3.RTZ(), value, z3.BitVecSort(BITS)), "i")
        self.defined.append(
            z3.And(z3.fpGT(value, _fp(-1.0)), z3.fpLT(value, _fp(2.0**32)))
        )
        return Value(z3.fpToUBV(z3.RTZ(), value, z3.BitVecSort(BITS)), "u")

    def arithmetic(self, op, left, right) -> Value:
        if not {left.kind, right.kind} <= {"i", "u"}:
            raise Unmodelled(f"{left.kind!r} and {right.kind!r} operands of {op}")
        # Usual arithmetic conversions: a signed and an unsigned 32-bit operand compute as unsigned.
        kind = "u" if "u" in (left.kind, right.kind) else "i"
        a, b = left.term, right.term
        if kind == "i" and not self.signed_wraps:
            self.defined.append(_no_signed_overflow(op, a, b))
        result = {"+": a + b, "-": a - b, "*": a * b}[op]
        return Value(result, kind)

    def division(self, op, left, right) -> Value:
        if not {left.kind, right.kind} <= {"i", "u"}:
            raise Unmodelled(f"{left.kind!r} and {right.kind!r} operands of {op}")
        if "u" not in (left.kind, right.kind):
            raise Unmodelled(f"signed {op}")
        a, b = left.term, right.term
        exact = z3.UDiv(a, b) if op == "/" else z3.URem(a, b)
        zero = b == 0
        behaviour = UNSIGNED_DIVISION_BY_ZERO.get(self.target, "undefined")
        if behaviour == "undefined":
            self.defined.append(z3.Not(zero))
            return Value(exact, "u")
        if behaviour == "unspecified":
            unspecified = z3.FreshConst(z3.BitVecSort(BITS), "unspecified")
            return Value(z3.If(zero, unspecified, exact), "u")
        by_zero = a if op == "/" else z3.BitVecVal(0, BITS)
        return Value(z3.If(zero, by_zero, exact), "u")


def _no_signed_overflow(op, a, b):
    if op == "+":
        return z3.And(z3.BVAddNoOverflow(a, b, True), z3.BVAddNoUnderflow(a, b))
    if op == "-":
        return z3.And(z3.BVSubNoOverflow(a, b), z3.BVSubNoUnderflow(a, b, True))
    return z3.And(z3.BVMulNoOverflow(a, b, True), z3.BVMulNoUnderflow(a, b))


def carried_wrap(op, operands):
    """The wrap rule: the low-order 32 bits of the exact result, defined for every input."""
    if op == "-" and len(operands) == 1:
        return -operands[0]
    a, b = operands
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def carried_udiv_zero(op, operands):
    """The Direct3D rule: the unsigned quotient or remainder, and all ones for a zero divisor."""
    a, b = operands
    exact = z3.UDiv(a, b) if op == "/" else z3.URem(a, b)
    return z3.If(b == 0, z3.BitVecVal(ALL_ONES, BITS), exact)


def carried_convert_sat(op, operands):
    """OpenCL C 6.4.3.3 saturated conversion: NaN gives 0, other values truncate and clamp to the result range.

    Stated over IEEE binary32 values rather than bit patterns, so it is independent of the helpers' bit tests.
    """
    (x,) = operands
    int_min, int_max = 1 << (BITS - 1), (1 << (BITS - 1)) - 1
    if op == "itou":
        return z3.If(x < 0, z3.BitVecVal(0, BITS), x)
    if op == "utoi":
        return z3.If(z3.UGT(x, int_max), z3.BitVecVal(int_max, BITS), x)
    value = z3.fpBVToFP(x, z3.Float32())
    zero = z3.BitVecVal(0, BITS)
    if op == "ftoi":
        clamped = z3.If(
            z3.fpGEQ(value, _fp(2.0**31)),
            z3.BitVecVal(int_max, BITS),
            z3.If(
                z3.fpLEQ(value, _fp(-(2.0**31))),
                z3.BitVecVal(int_min, BITS),
                z3.fpToSBV(z3.RTZ(), value, z3.BitVecSort(BITS)),
            ),
        )
    else:
        clamped = z3.If(
            z3.fpGEQ(value, _fp(2.0**32)),
            z3.BitVecVal(ALL_ONES, BITS),
            z3.If(
                z3.fpLEQ(value, _fp(0.0)),
                zero,
                z3.fpToUBV(z3.RTZ(), value, z3.BitVecSort(BITS)),
            ),
        )
    return z3.If(z3.fpIsNaN(value), zero, clamped)


def _fp(value):
    return z3.FPVal(value, z3.Float32())


_ORDERINGS = {
    "<": lambda a, b, signed: a < b if signed else z3.ULT(a, b),
    "<=": lambda a, b, signed: a <= b if signed else z3.ULE(a, b),
    ">": lambda a, b, signed: a > b if signed else z3.UGT(a, b),
    ">=": lambda a, b, signed: a >= b if signed else z3.UGE(a, b),
}

CARRIED = {
    rule_table.WRAP: carried_wrap,
    rule_table.UDIV_ZERO: carried_udiv_zero,
    rule_table.CONVERT_SAT: carried_convert_sat,
}
# The operand and result kind of each family's operations; conversions have their own signatures.
FAMILY_KIND = {rule_table.WRAP: "i", rule_table.UDIV_ZERO: "u"}
_SCALAR_KIND = {"float": "f", "int": "i", "uint": "u"}


def signature(intrinsic):
    """The operand kinds and the result kind of ``intrinsic``."""
    arity = rule_table.INTRINSICS[intrinsic][1]
    conversion = rule_table.CONVERSION_SIGNATURES.get(intrinsic)
    if conversion is not None:
        operand, result = (_SCALAR_KIND[name] for name in conversion)
        return (operand,) * arity, result
    kind = FAMILY_KIND[rule_table.INTRINSIC_FAMILY[intrinsic]]
    return (kind,) * arity, kind


def native_expression(intrinsic):
    operator, arity = rule_table.INTRINSICS[intrinsic]
    operands = [A.IdentifierNode(f"x{index}") for index in range(arity)]
    conversion = rule_table.CONVERSION_SIGNATURES.get(intrinsic)
    if conversion is not None:
        return A.FunctionCallNode(A.IdentifierNode(conversion[1]), operands)
    if arity == 1:
        return A.UnaryOpNode(operator, operands[0])
    return A.BinaryOpNode(operands[0], operator, operands[1])


def parameter_uses(expression):
    counts = {}
    for node in expression.walk():
        if isinstance(node, A.IdentifierNode):
            counts[node.name] = counts.get(node.name, 0) + 1
    return counts


def check_entry(intrinsic, target, entry):
    """Check one (intrinsic, target) entry; returns a result record."""
    record = {"intrinsic": intrinsic, "target": target, "kind": entry.kind}
    if entry.kind == "refuse":
        record["result"] = "refused"
        return record
    operator, arity = rule_table.INTRINSICS[intrinsic]
    family = rule_table.INTRINSIC_FAMILY[intrinsic]
    operand_kinds, kind = signature(intrinsic)
    if entry.kind == "polyfill" and intrinsic in rule_table.HELPERS:
        expression = helper_body(intrinsic)
    elif entry.kind == "polyfill":
        expression = template(intrinsic)
        uses = parameter_uses(expression)
        expected = {f"x{index}": 1 for index in range(arity)}
        if {k: v for k, v in uses.items() if k.startswith("x")} != expected:
            record["result"] = "malformed"
            record["detail"] = (
                f"parameter uses {uses}, expected each of x0..x{arity - 1} once"
            )
            return record
    elif entry.kind == "native":
        expression = native_expression(intrinsic)
    else:
        record["result"] = "malformed"
        record["detail"] = f"unknown entry kind {entry.kind!r}"
        return record
    operands = [z3.BitVec(f"x{index}", BITS) for index in range(arity)]
    semantics = Semantics(
        target,
        entry.configuration,
        {
            f"x{i}": Value(term, operand_kind)
            for i, (term, operand_kind) in enumerate(zip(operands, operand_kinds))
        },
    )
    try:
        result = semantics.eval(expression)
    except Unmodelled as error:
        record["result"] = "unmodelled"
        record["detail"] = str(error)
        return record
    if result.kind != kind:
        record["result"] = "malformed"
        record["detail"] = f"the construct produces {result.kind!r}, not {kind!r}"
        return record
    # Contextual simplification first: a guarded division such as ``b == 0 ? c : a / (b == 0 ? 1 : b)`` reduces to
    # the carried rule's own ``a / b`` term, which plain bit-blasting of two 32-bit dividers cannot prove in time.
    # Every tactic in the pipeline preserves equivalence, so unsat stays a proof and a model stays a counterexample.
    solver = z3.Then("simplify", "ctx-simplify", "simplify", "smt").solver()
    solver.set("timeout", 60000)
    refines = z3.And(
        *semantics.defined, result.term == CARRIED[family](operator, operands)
    )
    solver.add(z3.Not(refines))
    verdict = solver.check()
    if verdict == z3.unsat:
        record["result"] = "sound"
    elif verdict == z3.sat:
        model = solver.model()
        record["result"] = "unsound"
        record["counterexample"] = {
            str(x): hex(model.eval(x, model_completion=True).as_long())
            for x in operands
        }
    else:
        record["result"] = "unknown"
    if entry.configuration:
        record["configuration"] = entry.configuration
    return record


def check_table(entries=None, backends=None):
    """Every entry for every family, intrinsic and registered backend; returns (records, failures)."""
    if entries is None:
        entries = rule_table.TARGET_ENTRIES
    if backends is None:
        from crosstl.translator.codegen.registry import backend_names

        backends = backend_names()
    records, failures = [], []
    for family in rule_table.FAMILIES:
        for target in sorted(backends):
            entry = entries.get((family, target))
            if entry is None:
                failure = {"family": family, "target": target, "result": "missing"}
                records.append(failure)
                failures.append(failure)
                continue
            for intrinsic in sorted(rule_table.INTRINSICS):
                if rule_table.INTRINSIC_FAMILY[intrinsic] != family:
                    continue
                record = check_entry(intrinsic, target, entry)
                records.append(record)
                if record["result"] not in ("sound", "refused"):
                    failures.append(record)
    for intrinsic in sorted(rule_table.INTRINSICS):
        if (
            intrinsic not in rule_table.POLYFILLS
            and intrinsic not in rule_table.HELPERS
        ):
            failure = {"intrinsic": intrinsic, "result": "missing-polyfill"}
            records.append(failure)
            failures.append(failure)
    return records, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", action="store_true", help="Print every record as JSON"
    )
    args = parser.parse_args(argv)
    records, failures = check_table()
    if args.json:
        print(json.dumps(records, indent=1, sort_keys=True))
    else:
        for record in records:
            name = record.get("intrinsic", record.get("family", ""))
            extra = record.get("counterexample") or record.get("detail") or ""
            print(
                f"{name:16s} {record.get('target', ''):8s} {record.get('kind', ''):9s} "
                f"{record['result']:10s} {extra}"
            )
        print(f"{len(records)} records, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

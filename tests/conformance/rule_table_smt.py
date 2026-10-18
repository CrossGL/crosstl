"""SMT check of the rule table (crosstl/translator/semantics/rule_table.py) with Z3.

For every native and polyfill entry, ask Z3 for operands on which the target construct does not refine the carried
rule, in the style of Alive2:

    find x :  not ( defined_T(x) and result_T(x) == rule_S(x) )

``rule_S`` is the carried rule; for ``wrap`` it is the low-order 32 bits of the exact result, defined for every
input. The target construct is not modelled by hand: a native entry is the hub operation ``x0 op x1`` and a
polyfill entry is the template the lowering pass instantiates, parsed by the CrossGL parser, and both are evaluated
symbolically under the target's semantics of its trusted base (below). Satisfiable means unsound and comes with a
counterexample; unsatisfiable means sound under the model.

Target model:
  - unsigned 32-bit ``+``, ``-``, ``*`` are modular on every target (MSL 4.1 section 3.1, C++17 [basic.fundamental]
    for CUDA, HIP and Slang's C-like targets, GLSL 4.60 section 4.1.3, SPIR-V OpIAdd, WGSL section 6.2.3, Mojo
    UInt32 by reading);
  - signed ``+``, ``-``, ``*`` and unary ``-`` wrap on GLSL, GLSL ES and WGSL; they are undefined on overflow on
    Metal, DirectX (DXC marks them nsw), CUDA, HIP, Slang and Mojo, and on Vulkan, where the SPIR-V rule wraps but
    MoltenVK and KosmicKrisp hand the operation to MSL; on Rust they wrap under ``overflow-checks = false``, the
    configuration the entry records, and panic otherwise;
  - ``asint`` and ``asuint`` reinterpret the 32 bits exactly, and integer literals are exact.

The check also fails when an entry is missing for a registered backend, when a template uses a parameter other than
exactly once, or when a construct of a template has no model here. Run ``python -m tests.conformance.rule_table_smt``;
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
from crosstl.translator.semantics.hub_types import call_name, type_from_node
from crosstl.translator.semantics.lower import template

BITS = 32
SIGNED_WRAPS = frozenset({"opengl", "webgl", "wgsl"})
# A configuration under which a target's signed arithmetic wraps; the entry must record it.
SIGNED_WRAPS_UNDER = {"rust": "overflow-checks = false"}


class Unmodelled(Exception):
    """A template construct the checker has no semantics for."""


@dataclass(frozen=True)
class Value:
    term: object
    kind: str  # "i" or "u"


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
                name not in ("asint", "asuint", "int", "uint")
                or len(node.arguments) != 1
            ):
                raise Unmodelled(f"call {name!r}")
            operand = self.eval(node.arguments[0])
            return Value(operand.term, "i" if name in ("asint", "int") else "u")
        if isinstance(node, A.BinaryOpNode) and node.op in ("+", "-", "*"):
            return self.arithmetic(node.op, self.eval(node.left), self.eval(node.right))
        if isinstance(node, A.UnaryOpNode) and node.op == "-":
            operand = self.eval(node.operand)
            if operand.kind == "i" and not self.signed_wraps:
                self.defined.append(z3.BVSNegNoOverflow(operand.term))
            return Value(-operand.term, operand.kind)
        raise Unmodelled(f"{type(node).__name__} {getattr(node, 'op', '')}".strip())

    def arithmetic(self, op, left, right) -> Value:
        # Usual arithmetic conversions: a signed and an unsigned 32-bit operand compute as unsigned.
        kind = "u" if "u" in (left.kind, right.kind) else "i"
        a, b = left.term, right.term
        if kind == "i" and not self.signed_wraps:
            self.defined.append(_no_signed_overflow(op, a, b))
        result = {"+": a + b, "-": a - b, "*": a * b}[op]
        return Value(result, kind)


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


def native_expression(intrinsic):
    operator, arity = rule_table.WRAP_INTRINSICS[intrinsic]
    operands = [A.IdentifierNode(f"x{index}") for index in range(arity)]
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
    operator, arity = rule_table.WRAP_INTRINSICS[intrinsic]
    if entry.kind == "polyfill":
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
        {f"x{i}": Value(term, "i") for i, term in enumerate(operands)},
    )
    try:
        result = semantics.eval(expression)
    except Unmodelled as error:
        record["result"] = "unmodelled"
        record["detail"] = str(error)
        return record
    if result.kind != "i":
        record["result"] = "malformed"
        record["detail"] = "the construct does not produce a signed int"
        return record
    solver = z3.Solver()
    solver.set("timeout", 60000)
    refines = z3.And(
        *semantics.defined, result.term == carried_wrap(operator, operands)
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
            for intrinsic in sorted(rule_table.WRAP_INTRINSICS):
                if rule_table.INTRINSIC_FAMILY[intrinsic] != family:
                    continue
                record = check_entry(intrinsic, target, entry)
                records.append(record)
                if record["result"] not in ("sound", "refused"):
                    failures.append(record)
    for intrinsic in rule_table.WRAP_INTRINSICS:
        if intrinsic not in rule_table.POLYFILLS:
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

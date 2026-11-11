"""Affine congruences between integer locals, for relational index bounds.

After ``uint c = y; y = uint(n - 1 - y);`` the locals satisfy
``y == n - 1 - c (mod 2**32)`` until either is written again. Interval facts
keep one range per name, so a guard ``c < n`` bounds ``c`` but not ``y``, and
``in[y * stride + i]`` stays unproven. This module recovers that relation.

One pass over a function records, at every read of a tracked integer local,
each tracked local's value as an affine form over symbols, valid modulo
``2**bits`` of that local. A symbol is either the value an expression had where
it was evaluated, bounded by the codegen's interval facts at that expression,
or an unknown value introduced where control flow merges, a loop iterates or
an untracked write occurs; unknown values must cancel. Ring arithmetic
(``+``, ``-``, unary ``-`` and ``*`` by a literal) at least ``bits`` wide, and
integer conversions to at least ``bits`` bits, preserve values modulo
``2**bits``; everything else becomes a symbol.

A read is bounded by rewriting its form as a signed sum of other tracked
locals' forms plus a remainder free of unknown values, and evaluating that sum
with the interval facts at the read. When the sum's range lies within the
read local's own domain, the congruence is an equality, so the range bounds
the read. Every complete rewrite is sound; the bound is their intersection.

Interval facts also lose the difference a guard such as ``c < n`` states when
``n`` is not a constant: from intervals alone, ``n - 1 - c`` may be negative,
although the guard makes it non-negative. Each comparison conjunct of an
enclosing ``if`` condition is a linear fact ``E <= 0`` over exact values when
its operands are tracked locals, immutable parameters or integer literals that
compare exactly (the usual arithmetic conversions keep both values). A fact
about a local holds at a read while that local's form is unchanged since the
guard. Every read of an immutable parameter (by value, or a read-only
``constant`` reference, never written) has the same value, so their symbols
share one term, bounded by the intersection of the bounds at those reads; each
of them ran on the path to the read. A fact bounds the sum ``S`` of a rewrite
by ``min(S + E) <= S <= max(S - E)``.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Any

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    CaseNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    ReferenceType,
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .constant_guards import literal_guard_value

# Bounds the rewrite search: at most this many related locals, used at most
# ``MAX_REWRITE_TERMS`` at a time.
MAX_RELATED_LOCALS = 6
MAX_REWRITE_TERMS = 3
INCREMENTS = {"++": 1, "--": -1}
# ``a OP b`` as facts ``sign_a * a + sign_b * b + constant <= 0``.
COMPARISON_FACTS = {
    "<": ((1, -1, 1),),
    "<=": ((1, -1, 0),),
    ">": ((-1, 1, 1),),
    ">=": ((-1, 1, 0),),
    "==": ((1, -1, 0), (-1, 1, 0)),
}
INCREMENT_TOKENS = {
    "PRE_INCREMENT": 1,
    "POST_INCREMENT": 1,
    "PRE_DECREMENT": -1,
    "POST_DECREMENT": -1,
}


def _increment(codegen: Any, node: Any) -> int | None:
    if not isinstance(node, UnaryOpNode):
        return None
    operator = getattr(node, "operator", getattr(node, "op", None))
    if operator in INCREMENT_TOKENS:
        return INCREMENT_TOKENS[operator]
    return INCREMENTS.get(codegen.map_operator(operator))


def _children(node: Any):
    if isinstance(node, (list, tuple)):
        yield from node
    elif isinstance(node, dict):
        yield from node.values()
    elif hasattr(node, "child_nodes"):
        yield from node.child_nodes()


def _walk(node: Any):
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        if id(current) in seen:
            continue
        seen.add(id(current))
        if not isinstance(current, (list, tuple, dict)):
            yield current
        stack.extend(_children(current))


def _root(node: Any) -> Any:
    while True:
        if isinstance(node, MemberAccessNode):
            node = node.object
        elif isinstance(node, ArrayAccessNode):
            node = node.array
        elif isinstance(node, UnaryOpNode) and node.op in {"&", "*"}:
            node = node.operand
        else:
            return node if isinstance(node, IdentifierNode) else None


def _form_add(left, right, scale=1):
    coefficients = dict(left[0])
    for symbol, value in right[0]:
        coefficients[symbol] = coefficients.get(symbol, 0) + scale * value
    return (
        tuple(sorted((s, v) for s, v in coefficients.items() if v)),
        left[1] + scale * right[1],
    )


def _form_scale(form, factor):
    return (
        tuple((s, v * factor) for s, v in form[0] if v * factor),
        form[1] * factor,
    )


def _exact_comparison(left, right) -> bool:
    """Whether the usual arithmetic conversions keep both operands' values."""
    if left[2] is None or right[2] is None:
        literal, other = (left, right) if left[2] is None else (right, left)
        return other[2] is None or literal[1] >= 0 or other[2][1]
    (left_bits, left_signed), (right_bits, right_signed) = left[2], right[2]
    if left_signed == right_signed:
        return True
    signed_bits = left_bits if left_signed else right_bits
    unsigned_bits = right_bits if left_signed else left_bits
    return signed_bits > unsigned_bits


class _Relations:
    """Affine forms of tracked locals at every read of one function."""

    def __init__(self, codegen: Any, function: Any):
        self.codegen = codegen
        lexical = codegen.glsl_scalar_array_view_lexical_bindings(function)
        self.declarations = lexical["declarations"]
        self.references = lexical["reference_bindings"]
        counts: dict[str, int] = {}
        for declaration in self.declarations.values():
            counts[declaration["name"]] = counts.get(declaration["name"], 0) + 1
        self.source_types = {
            declaration["name"]: declaration["source_type"]
            for declaration in self.declarations.values()
            if declaration.get("source_type") is not None
            and counts[declaration["name"]] == 1
        }
        self.unique = {name for name, count in counts.items() if count == 1}
        body = getattr(function, "body", None)
        escaped = self._escaped_names(body)
        self.tracked: dict[int, tuple[int, bool, str]] = {}
        for key, declaration in self.declarations.items():
            name = declaration["name"]
            if name not in self.unique or name in escaped:
                continue
            domain = self._scalar_width(declaration.get("source_type"))
            if domain is not None:
                self.tracked[key] = (*domain, name)
        # Symbol ``i`` is the value of ``self.symbols[i]`` where it was
        # evaluated, or an unknown value when that entry is ``None``.
        self.symbols: list[Any] = []
        # Symbol ``i`` reads the immutable parameter ``self.aliases[i]``.
        self.aliases: dict[int, int] = {}
        self.immutable = self._immutable_parameters(body, escaped)
        # Facts of the enclosing guards, innermost last.
        self.facts: list[tuple] = []
        self.reads: dict[int, tuple[dict, int, tuple]] = {}
        self.bounds_cache: dict[int, Any] = {}
        entry = {}
        for key, declaration in self.declarations.items():
            if key in self.tracked and declaration["kind"] == "parameter":
                entry[key] = self._unknown()
        with codegen.glsl_lexical_source_type_scope(self.source_types):
            self._statement(body, entry)

    # Types and symbols -------------------------------------------------------

    def _scalar_width(self, type_name: Any) -> tuple[int, bool] | None:
        text = self.codegen.type_name_string(type_name) or ""
        if "*" in text or "&" in text:
            return None
        info = self.codegen.glsl_value_type_info(type_name)
        if info is None or info["width"] != 1 or info["family"] not in {"int", "uint"}:
            return None
        bits = info["narrow"]["bits"] if info["narrow"] is not None else info["bits"]
        return bits, info["family"] == "int"

    def _value_domain(self, type_name: Any) -> tuple[int, bool] | None:
        """``_scalar_width`` of a value, or of a read-only constant reference."""
        if isinstance(type_name, ReferenceType):
            if type_name.is_mutable or type_name.address_space != "constant":
                return None
            type_name = type_name.referenced_type
        return self._scalar_width(type_name)

    def _immutable_parameters(self, body: Any, escaped: set[str]) -> set[int]:
        """Untracked integer parameters that no statement can change."""
        written: set[Any] = set()
        for node in _walk(body):
            if isinstance(node, AssignmentNode):
                targets = [getattr(node, "target", getattr(node, "left", None))]
            elif _increment(self.codegen, node) is not None:
                targets = [node.operand]
            elif (
                isinstance(node, FunctionCallNode)
                and self.codegen.glsl_integer_scalar_cast_call(node) is None
            ):
                targets = list(getattr(node, "arguments", None) or [])
            else:
                continue
            for target in targets:
                root = _root(target)
                if root is not None:
                    written.add(self.references.get(id(root)))
        return {
            key
            for key, declaration in self.declarations.items()
            if declaration["kind"] == "parameter"
            and key not in self.tracked
            and key not in written
            and declaration["name"] in self.unique
            and declaration["name"] not in escaped
            and self._value_domain(declaration.get("source_type")) is not None
        }

    def _escaped_names(self, body: Any) -> set[str]:
        """Locals another name may write: address-taken or bound by reference."""
        names: set[str] = set()
        for node in _walk(body):
            if (
                isinstance(node, UnaryOpNode)
                and self.codegen.map_operator(node.op) == "&"
            ):
                root = _root(node.operand)
                if root is not None:
                    names.add(root.name)
            elif isinstance(node, (VariableNode, ArrayNode)):
                text = self.codegen.type_name_string(
                    getattr(node, "var_type", getattr(node, "vtype", None))
                )
                if text and ("&" in text or "*" in text):
                    names.add(node.name)
                    root = _root(getattr(node, "initial_value", None))
                    if root is not None:
                        names.add(root.name)
        return names

    def _unknown(self):
        self.symbols.append(None)
        return ((len(self.symbols) - 1, 1),), 0

    def _value(self, node: Any):
        self.symbols.append(node)
        if isinstance(node, IdentifierNode):
            key = self.references.get(id(node))
            if key in self.immutable:
                self.aliases[len(self.symbols) - 1] = key
        return ((len(self.symbols) - 1, 1),), 0

    def _kill(self, env: dict, keys: set[int]) -> dict:
        if not keys:
            return env
        result = dict(env)
        for key in sorted(keys, key=lambda k: self.tracked[k][2]):
            result[key] = self._unknown()
        return result

    # Writes ------------------------------------------------------------------

    def _written(self, region: Any) -> set[int]:
        """Tracked locals any write or call argument in ``region`` may change."""
        keys: set[int] = set()

        def add(node: Any) -> None:
            root = _root(node)
            if root is None:
                return
            key = self.references.get(id(root))
            if key in self.tracked:
                keys.add(key)

        for node in _walk(region):
            if isinstance(node, AssignmentNode):
                add(getattr(node, "target", getattr(node, "left", None)))
            elif _increment(self.codegen, node) is not None:
                add(node.operand)
            elif isinstance(node, (VariableNode, ArrayNode)):
                if id(node) in self.tracked:
                    keys.add(id(node))
            elif isinstance(node, FunctionCallNode):
                if self.codegen.glsl_integer_scalar_cast_call(node) is not None:
                    continue
                for argument in getattr(node, "arguments", None) or []:
                    add(argument)
        return keys

    def _has_writes(self, expression: Any) -> bool:
        for node in _walk(expression):
            if isinstance(node, AssignmentNode) or _increment(self.codegen, node):
                return True
            if (
                isinstance(node, FunctionCallNode)
                and self.codegen.glsl_integer_scalar_cast_call(node) is None
            ):
                return True
        return False

    # Expressions -------------------------------------------------------------

    def _record(self, expression: Any, env: dict) -> None:
        for node in _walk(expression):
            if isinstance(node, IdentifierNode):
                key = self.references.get(id(node))
                if key in self.tracked and key in env:
                    self.reads[id(node)] = (env, key, tuple(self.facts))

    def _expression(self, expression: Any, env: dict) -> dict:
        """Record the reads in ``expression`` and return the state after it."""
        if expression is None or isinstance(expression, (str, int, float, bool)):
            return env
        if not self._has_writes(expression):
            self._record(expression, env)
            return env
        written = self._written(expression)
        # Reads may happen before or after the writes inside the expression.
        self._record(expression, self._kill(env, written))
        return self._kill(env, written)

    def _result_width(self, node: Any) -> int | None:
        domain = self._scalar_width(self.codegen.glsl_source_expression_type(node))
        return None if domain is None else domain[0]

    def _ambiguous(self, node: Any) -> bool:
        """Whether a name in ``node`` refers to a shadowed declaration."""
        for child in _walk(node):
            if isinstance(child, IdentifierNode):
                key = self.references.get(id(child))
                if (
                    key is not None
                    and self.declarations[key]["name"] not in self.unique
                ):
                    return True
        return False

    def _form(self, node: Any, bits: int, env: dict):
        """The value of ``node`` modulo ``2**bits`` as an affine form."""
        if isinstance(node, LiteralNode):
            literal = self.codegen.literal_int_value(node, {})
            if isinstance(literal, int) and not isinstance(literal, bool):
                return (), literal
            return self._value(node)
        if isinstance(node, IdentifierNode):
            key = self.references.get(id(node))
            if key in self.tracked and key in env and self.tracked[key][0] >= bits:
                return env[key]
            return self._value(node)
        if self._ambiguous(node):
            return self._value(node)
        cast = self.codegen.glsl_integer_scalar_cast_call(node)
        if cast is not None:
            destination, (argument,) = cast
            domain = self._scalar_width(destination)
            if domain is not None and domain[0] >= bits:
                return self._form(argument, bits, env)
            return self._value(node)
        width = (
            self._result_width(node)
            if isinstance(node, (UnaryOpNode, BinaryOpNode))
            else None
        )
        if width is None or width < bits:
            return self._value(node)
        operator = self.codegen.map_operator(node.op)
        if isinstance(node, UnaryOpNode) and operator in {"+", "-"}:
            inner = self._form(node.operand, bits, env)
            return inner if operator == "+" else _form_scale(inner, -1)
        if isinstance(node, BinaryOpNode) and operator in {"+", "-", "*"}:
            left = self._form(node.left, bits, env)
            right = self._form(node.right, bits, env)
            if operator != "*":
                return _form_add(left, right, 1 if operator == "+" else -1)
            for factor, other in ((left, right), (right, left)):
                if not factor[0]:
                    return _form_scale(other, factor[1])
        return self._value(node)

    # Statements --------------------------------------------------------------

    def _assign(self, key: int, value: Any, operator: str, env: dict) -> dict:
        bits = self.tracked[key][0]
        if operator == "=":
            form = self._form(value, bits, env)
        elif operator in {"+=", "-="} and key in env:
            form = _form_add(
                env[key],
                self._form(value, bits, env),
                1 if operator == "+=" else -1,
            )
        elif operator == "*=" and key in env:
            factor = self._form(value, bits, env)
            form = _form_scale(env[key], factor[1]) if not factor[0] else None
        else:
            form = None
        result = dict(env)
        result[key] = form if form is not None else self._unknown()
        return result

    def _simple_write(self, node: Any, env: dict) -> dict | None:
        """Transfer ``x = e``, ``x op= e`` and ``x++`` on a tracked local."""
        if isinstance(node, AssignmentNode):
            target = getattr(node, "target", getattr(node, "left", None))
            value = getattr(node, "value", getattr(node, "right", None))
            key = (
                self.references.get(id(target))
                if isinstance(target, IdentifierNode)
                else None
            )
            if key not in self.tracked or self._has_writes(value):
                return None
            self._record(value, env)
            operator = self.codegen.map_operator(
                getattr(node, "operator", getattr(node, "op", "="))
            )
            return self._assign(key, value, operator, env)
        step = _increment(self.codegen, node)
        if step is not None and isinstance(node.operand, IdentifierNode):
            key = self.references.get(id(node.operand))
            if key in self.tracked and key in env:
                result = dict(env)
                result[key] = _form_add(env[key], ((), step))
                return result
        return None

    def _sequence(self, statements: Any, env: dict) -> dict:
        for statement in statements:
            env = self._statement(statement, env)
        return env

    @staticmethod
    def _scoped(outer: dict, inner: dict) -> dict:
        """Drop the locals a nested scope declared once it ends."""
        return {key: form for key, form in inner.items() if key in outer}

    def _join(self, envs: list[dict]) -> dict:
        keys = set().union(*envs)
        result = {}
        for key in sorted(keys, key=lambda k: self.tracked[k][2]):
            forms = [env.get(key) for env in envs]
            first = forms[0]
            result[key] = (
                first
                if first is not None and all(form == first for form in forms)
                else self._unknown()
            )
        return result

    def _if(self, node: Any, env: dict) -> dict:
        condition = getattr(node, "if_condition", node.condition)
        env = self._expression(condition, env)
        taken = literal_guard_value(condition, {}, self.codegen.map_operator)
        then_body = getattr(node, "if_body", None)
        else_ifs = list(zip(node.else_if_conditions or [], node.else_if_bodies or []))
        else_body = getattr(node, "else_body", None)
        if taken is True:
            return self._guarded(then_body, condition, env)
        outcomes = [] if taken is False else [self._guarded(then_body, condition, env)]
        for else_condition, body in else_ifs:
            env = self._expression(else_condition, env)
            outcomes.append(self._guarded(body, else_condition, env))
        outcomes.append(self._statement(else_body, env))
        return self._join(outcomes)

    def _guarded(self, body: Any, condition: Any, env: dict) -> dict:
        """Transfer ``body``, which runs only where ``condition`` held."""
        facts = self._guard_facts(condition, env)
        depth = len(self.facts)
        self.facts.extend(facts)
        try:
            return self._statement(body, env)
        finally:
            del self.facts[depth:]

    def _fact_operand(self, node: Any, env: dict):
        """``(terms, constant, domain)`` of an exact comparison operand."""
        if isinstance(node, LiteralNode):
            literal = self.codegen.literal_int_value(node, {})
            if isinstance(literal, int) and not isinstance(literal, bool):
                return (), literal, None
            return None
        if not isinstance(node, IdentifierNode):
            return None
        key = self.references.get(id(node))
        if key in self.tracked and key in env:
            return ((("local", key, env[key]), 1),), 0, self.tracked[key][:2]
        if key in self.immutable:
            ((symbol, _),), _ = self._value(node)
            domain = self._value_domain(self.declarations[key].get("source_type"))
            return ((("symbol", symbol), 1),), 0, domain
        return None

    def _guard_facts(self, condition: Any, env: dict) -> list[tuple]:
        """Facts ``(terms, constant)``, meaning ``sum + constant <= 0``."""
        if condition is None or self._has_writes(condition):
            return []
        facts, pending = [], [condition]
        while pending:
            node = pending.pop()
            if not isinstance(node, BinaryOpNode):
                continue
            operator = self.codegen.map_operator(node.op)
            if operator == "&&":
                pending.extend((node.right, node.left))
                continue
            if operator not in COMPARISON_FACTS:
                continue
            left = self._fact_operand(node.left, env)
            right = self._fact_operand(node.right, env)
            if left is None or right is None or not _exact_comparison(left, right):
                continue
            for left_sign, right_sign, constant in COMPARISON_FACTS[operator]:
                terms: dict = {}
                for operand, sign in ((left, left_sign), (right, right_sign)):
                    for term, coefficient in operand[0]:
                        terms[term] = terms.get(term, 0) + sign * coefficient
                facts.append(
                    (
                        tuple((term, value) for term, value in terms.items() if value),
                        left_sign * left[1] + right_sign * right[1] + constant,
                    )
                )
        return facts

    def _loop(self, node: Any, env: dict) -> dict:
        outer = env
        init = getattr(node, "init", None)
        env = self._statement(init, env)
        condition = getattr(node, "condition", None)
        update = getattr(node, "update", None)
        body = getattr(node, "body", None)
        written = self._written([condition, update, body])
        head = self._kill(env, written)
        if not isinstance(node, DoWhileNode):
            head = self._expression(condition, head)
        self._statement(body, head)
        self._expression(update, self._kill(env, written))
        if isinstance(node, DoWhileNode):
            self._expression(condition, self._kill(env, written))
        return self._scoped(outer, self._kill(env, written))

    def _switch(self, node: Any, env: dict) -> dict:
        env = self._expression(node.expression, env)
        cases = list(node.cases or [])
        written = self._written([cases, node.default_case])
        for case in cases:
            body = case.statements if isinstance(case, CaseNode) else case
            self._statement(body, self._kill(env, written))
        default = node.default_case
        if default is not None:
            body = default.statements if isinstance(default, CaseNode) else default
            self._statement(body, self._kill(env, written))
        return self._kill(env, written)

    def _statement(self, node: Any, env: dict) -> dict:
        if node is None or isinstance(node, (str, int, float, bool)):
            return env
        if isinstance(node, (list, tuple)):
            return self._sequence(node, env)
        if isinstance(node, BlockNode):
            return self._scoped(env, self._sequence(node.statements or [], env))
        if isinstance(node, (VariableNode, ArrayNode)):
            value = getattr(node, "initial_value", None)
            if id(node) not in self.tracked:
                return self._expression(value, env)
            if value is None:
                result = dict(env)
                result[id(node)] = self._unknown()
                return result
            if self._has_writes(value):
                env = self._expression(value, env)
                result = dict(env)
                result[id(node)] = self._value(value)
                return result
            self._record(value, env)
            return self._assign(id(node), value, "=", env)
        if isinstance(node, ExpressionStatementNode):
            return self._statement(node.expression, env)
        if isinstance(node, (AssignmentNode, UnaryOpNode)):
            written = self._simple_write(node, env)
            if written is not None:
                return written
            return self._expression(node, env)
        if isinstance(node, IfNode):
            return self._if(node, env)
        if isinstance(node, (ForNode, WhileNode, DoWhileNode)):
            return self._loop(node, env)
        if isinstance(node, SwitchNode):
            return self._switch(node, env)
        if isinstance(node, ReturnNode):
            return self._expression(node.value, env)
        if hasattr(node, "body") or hasattr(node, "statements"):
            # Other compound statements may run their parts in any order.
            written = self._written(node)
            self._record(node, self._kill(env, written))
            return self._kill(env, written)
        return self._expression(node, env)

    # Queries -----------------------------------------------------------------

    def _symbol_bounds(self, symbol: int):
        if symbol in self.bounds_cache:
            return self.bounds_cache[symbol]
        node = self.symbols[symbol]
        bounds = None
        if node is not None:
            value = self.codegen.glsl_index_value_range(node)
            if value is not None:
                bounds = (value.minimum, value.maximum)
        self.bounds_cache[symbol] = bounds
        return bounds

    def _term(self, symbol: int):
        """The term of a symbol: its parameter when immutable, else itself."""
        key = self.aliases.get(symbol)
        return ("symbol", symbol) if key is None else ("parameter", key)

    def _term_bounds(self, term, members: dict, local_bounds):
        if term[0] == "local":
            return local_bounds(term[1])
        found = None
        for symbol in sorted(members.get(term, ())):
            bounds = self._symbol_bounds(symbol)
            if bounds is not None:
                found = (
                    bounds
                    if found is None
                    else (max(found[0], bounds[0]), min(found[1], bounds[1]))
                )
        return found if found is None or found[0] <= found[1] else None

    def _sum_range(self, terms: dict, constant: int, members: dict, local_bounds):
        low = high = constant
        for term, coefficient in terms.items():
            if not coefficient:
                continue
            bounds = self._term_bounds(term, members, local_bounds)
            if bounds is None:
                return None
            low += coefficient * (bounds[0] if coefficient > 0 else bounds[1])
            high += coefficient * (bounds[1] if coefficient > 0 else bounds[0])
        return low, high

    def _usable_facts(self, env: dict, facts: tuple) -> list[tuple]:
        """Guard facts whose locals still hold their values from the guard."""
        usable = []
        for fact_terms, constant in facts:
            terms: dict = {}
            members: dict = {}
            for term, coefficient in fact_terms:
                if term[0] == "local":
                    if env.get(term[1]) != term[2]:
                        break
                    term = ("local", term[1])
                else:
                    symbol = term[1]
                    term = self._term(symbol)
                    members.setdefault(term, set()).add(symbol)
                terms[term] = terms.get(term, 0) + coefficient
            else:
                usable.append((terms, constant, members))
        return usable

    def _fact_range(self, terms, constant, members, facts, local_bounds):
        """The range of ``sum(terms) + constant``, tightened by ``facts``."""
        plain = self._sum_range(terms, constant, members, local_bounds)
        lows = [] if plain is None else [plain[0]]
        highs = [] if plain is None else [plain[1]]
        for fact_terms, fact_constant, fact_members in facts:
            merged = {
                term: members.get(term, set()) | fact_members.get(term, set())
                for term in set(members) | set(fact_members)
            }
            for sign, bounds in ((1, lows), (-1, highs)):
                combined = dict(terms)
                for term, coefficient in fact_terms.items():
                    combined[term] = combined.get(term, 0) + sign * coefficient
                found = self._sum_range(
                    combined, constant + sign * fact_constant, merged, local_bounds
                )
                if found is not None:
                    # E <= 0, so S + E <= S <= S - E.
                    bounds.append(found[0] if sign > 0 else found[1])
        if not lows or not highs:
            return None
        return max(lows), min(highs)

    def bound(self, node: Any, intervals: dict) -> tuple[int, int] | None:
        record = self.reads.get(id(node))
        if record is None:
            return None
        env, key, guard_facts = record
        bits, signed, _name = self.tracked[key]
        domain = (
            (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
            if signed
            else (0, (1 << bits) - 1)
        )
        form = env[key]
        symbols = {symbol for symbol, _ in form[0]}
        related = [
            other
            for other in sorted(env, key=lambda k: self.tracked[k][2])
            if other != key
            and self.tracked[other][0] >= bits
            and symbols & {symbol for symbol, _ in env[other][0]}
        ][:MAX_RELATED_LOCALS]
        facts = self._usable_facts(env, guard_facts)

        def local_bounds(other):
            interval = intervals.get(self.tracked[other][2])
            if interval is not None:
                return interval
            other_bits, other_signed, _ = self.tracked[other]
            if other_signed:
                return -(1 << (other_bits - 1)), (1 << (other_bits - 1)) - 1
            return 0, (1 << other_bits) - 1

        best = None
        for count in range(min(len(related), MAX_REWRITE_TERMS) + 1):
            for chosen in combinations(related, count):
                for signs in product((1, -1), repeat=count):
                    remainder = form
                    terms: dict = {}
                    for other, sign in zip(chosen, signs):
                        remainder = _form_add(remainder, env[other], -sign)
                        terms[("local", other)] = sign
                    members: dict = {}
                    for symbol, coefficient in remainder[0]:
                        if self.symbols[symbol] is None:
                            break
                        term = self._term(symbol)
                        terms[term] = terms.get(term, 0) + coefficient
                        members.setdefault(term, set()).add(symbol)
                    else:
                        found = self._fact_range(
                            terms, remainder[1], members, facts, local_bounds
                        )
                        if found is not None and (
                            domain[0] <= found[0] <= found[1] <= domain[1]
                        ):
                            best = (
                                found
                                if best is None
                                else (max(best[0], found[0]), min(best[1], found[1]))
                            )
        return best


def related_local_interval(codegen: Any, function: Any, node: Any, intervals: dict):
    """Bound a read of an integer local through its affine relations.

    Returns ``(minimum, maximum)`` or ``None``. ``intervals`` holds the
    interval facts at ``node``.
    """
    if function is None or not isinstance(node, IdentifierNode) or not intervals:
        return None
    cache = codegen.__dict__.setdefault("_glsl_affine_relations", {})
    entry = cache.get(id(function))
    if entry is None or entry[0] is not function:
        active = codegen.__dict__.setdefault("_glsl_affine_relations_active", set())
        if id(function) in active:
            return None
        active.add(id(function))
        try:
            entry = (function, _Relations(codegen, function), {})
        finally:
            active.discard(id(function))
        cache[id(function)] = entry
    _function, relations, results = entry
    if id(node) not in relations.reads:
        return None
    # Bounds depend on the interval facts the caller supplies.
    signature = (id(node), id(intervals))
    if signature in results:
        return results[signature][1]
    results[signature] = (intervals, None)
    bound = relations.bound(node, intervals)
    results[signature] = (intervals, bound)
    return bound

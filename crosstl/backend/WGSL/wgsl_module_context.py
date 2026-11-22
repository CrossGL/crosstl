"""Module-scope declarations of a WGSL module for the WGSL importer: structures, aliases, constants, overrides,
module-scope variables and functions, with their types resolved (:mod:`.wgsl_types`) and constant expressions
folded (W3C WGSL, section 8.5).

WGSL declares module-scope names in any order (section 5.1); CrossGL, like the languages it is written to, wants a
name declared before it is used. :meth:`ModuleContext.ordered` lists each kind of declaration so that every one
follows the declarations it names. WGSL forbids recursion and cyclic declarations (sections 5.1 and 11.1), so such
an order exists.
"""

from __future__ import annotations

import math

from . import WgslAst as A
from .wgsl_types import (
    ABSTRACT_FLOAT,
    ABSTRACT_INT,
    BOOL,
    F16,
    F32,
    I32,
    U32,
    Scalar,
    TypeResolver,
    WgslImportError,
)

__all__ = ["ModuleContext", "literal_value", "literal_type", "names_in"]

_INT_SUFFIX = {"i": I32, "u": U32}
_FLOAT_SUFFIX = {"f": F32, "h": F16}


def literal_type(node: A.Literal):
    """The type of a literal (section 3.4): an unsuffixed one is abstract."""
    if node.kind == "bool":
        return BOOL
    suffix = node.text[-1]
    if node.kind == "int":
        return _INT_SUFFIX.get(suffix, ABSTRACT_INT)
    if not node.text.lower().startswith("0x"):
        return _FLOAT_SUFFIX.get(suffix, ABSTRACT_FLOAT)
    # A hexadecimal float ends in a hexadecimal digit unless it has an exponent and a suffix.
    if "p" in node.text.lower():
        return _FLOAT_SUFFIX.get(suffix, ABSTRACT_FLOAT)
    return ABSTRACT_FLOAT


def literal_value(node: A.Literal):
    """The value of a literal: an ``int``, ``float`` or ``bool``."""
    text = node.text
    if node.kind == "bool":
        return text == "true"
    if node.kind == "int":
        digits = text[:-1] if text[-1] in "iu" else text
        return int(digits, 16) if digits.lower().startswith("0x") else int(digits)
    lowered = text.lower()
    if lowered.startswith("0x"):
        body = text[:-1] if ("p" in lowered and text[-1] in "fh") else text
        if "p" not in body.lower():
            body += "p0"
        return float.fromhex(body)
    return float(text[:-1] if text[-1] in "fh" else text)


def names_in(node, found=None) -> set:
    """Every identifier an expression, type or statement tree names."""
    found = set() if found is None else found
    if isinstance(node, A.Ident):
        found.add(node.name)
        for arg in node.template_args or ():
            names_in(arg, found)
    elif isinstance(node, list):
        for item in node:
            names_in(item, found)
    elif hasattr(node, "__dataclass_fields__"):
        for name in node.__dataclass_fields__:
            value = getattr(node, name)
            if isinstance(value, (list, A.Ident)) or hasattr(
                value, "__dataclass_fields__"
            ):
                names_in(value, found)
    return found


def _wrap32(value: int, signed: bool) -> int:
    value &= 0xFFFFFFFF
    if signed and value >= 0x80000000:
        value -= 0x100000000
    return value


class ModuleContext:
    """The declarations of one module, by name, with resolved types and folded constant values."""

    def __init__(self, module: A.Module):
        self.module = module
        self.enables = {
            name for d in module.directives if d.kind == "enable" for name in d.names
        }
        self.structs, self.aliases, self.consts, self.overrides = {}, {}, {}, {}
        self.globals, self.functions = {}, {}
        self._alias_nodes = {}
        for node in module.declarations:
            self._collect(node)
        self.resolver = TypeResolver(
            self.structs, self.aliases, self.eval_count, self.count_text
        )
        for name in self._ordered_names(self._alias_nodes, lambda n: n.type):
            self.aliases[name] = self.resolver.resolve(self._alias_nodes[name].type)
        self.struct_types = {
            name: [
                (m.name, self.resolver.resolve(m.type), m.attributes, m.line)
                for m in node.members
            ]
            for name, node in self.structs.items()
        }
        self.const_values = {}
        for name in self._ordered_names(self.consts, lambda n: n.init):
            self.const_values[name] = self.eval_const(self.consts[name].init)

    def _collect(self, node):
        table = {
            A.Struct: self.structs,
            A.Alias: self._alias_nodes,
            A.Const: self.consts,
            A.Override: self.overrides,
            A.GlobalVar: self.globals,
            A.Function: self.functions,
        }.get(type(node))
        if table is None:
            return
        if node.name in self.all_names():
            raise WgslImportError(
                f"'{node.name}' is declared twice at module scope", node.line
            )
        table[node.name] = node

    def all_names(self) -> set:
        return {
            *self.structs,
            *self._alias_nodes,
            *self.consts,
            *self.overrides,
            *self.globals,
            *self.functions,
        }

    # Ordering.

    def _ordered_names(self, table, uses):
        """The names of ``table`` with every one after the names of ``table`` it uses."""
        order, visiting, done = [], set(), set()

        def visit(name):
            if name in done:
                return
            if name in visiting:
                raise WgslImportError(
                    f"'{name}' is declared in terms of itself", table[name].line
                )
            visiting.add(name)
            for used in sorted(names_in(uses(table[name])) & set(table)):
                if used != name:
                    visit(used)
            visiting.discard(name)
            done.add(name)
            order.append(name)

        for name in table:
            visit(name)
        return order

    def ordered(self, table, uses):
        return [table[name] for name in self._ordered_names(table, uses)]

    def ordered_structs(self):
        return self.ordered(self.structs, lambda node: [m.type for m in node.members])

    def ordered_values(self):
        """Constants and overrides, each after the values it names."""
        values = {**self.consts, **self.overrides}
        return self.ordered(values, lambda node: [node.type, node.init])

    def ordered_functions(self):
        return self.ordered(self.functions, lambda node: node.body)

    # Constant expressions.

    def eval_count(self, node):
        """An array element count folded to an ``int``, or ``None`` for an override expression."""
        if names_in(node) & set(self.overrides):
            return None
        value = self.eval_const(node)
        if not isinstance(value, int) or isinstance(value, bool):
            raise WgslImportError(
                "an array count must be a constant integer", getattr(node, "line", None)
            )
        return value

    def count_text(self, node):
        from .wgsl_expressions import plain_expression_text

        return plain_expression_text(node)

    def eval_const(self, node):
        """The value of a constant expression, or ``None`` when it is not one this folder evaluates."""
        if isinstance(node, A.Literal):
            return literal_value(node)
        if isinstance(node, A.Ident):
            if node.template_args is None and node.name in self.const_values:
                return self.const_values[node.name]
            return None
        if isinstance(node, A.Unary):
            value = self.eval_const(node.operand)
            if value is None:
                return None
            return {"-": lambda v: -v, "!": lambda v: not v, "~": lambda v: ~v}.get(
                node.op, lambda v: None
            )(value)
        if isinstance(node, A.Binary):
            return self._eval_binary(node)
        if (
            isinstance(node, A.Call)
            and node.callee.name in ("i32", "u32", "f32")
            and len(node.args) == 1
        ):
            value = self.eval_const(node.args[0])
            if value is None or isinstance(value, bool):
                return None
            if node.callee.name == "f32":
                return float(value)
            return _wrap32(int(value), node.callee.name == "i32")
        return None

    def _eval_binary(self, node):
        left, right = self.eval_const(node.left), self.eval_const(node.right)
        if left is None or right is None:
            return None
        op = node.op
        try:
            if op in ("/", "%") and isinstance(left, int) and isinstance(right, int):
                if right == 0:
                    return None
                quotient = (
                    abs(left) // abs(right) * (1 if (left >= 0) == (right >= 0) else -1)
                )
                return quotient if op == "/" else left - right * quotient
            if op == "%":
                return math.fmod(left, right)
            return {
                "+": lambda: left + right,
                "-": lambda: left - right,
                "*": lambda: left * right,
                "/": lambda: left / right,
                "<<": lambda: left << right,
                ">>": lambda: left >> right,
                "&": lambda: left & right,
                "|": lambda: left | right,
                "^": lambda: left ^ right,
                "==": lambda: left == right,
                "!=": lambda: left != right,
                "<": lambda: left < right,
                ">": lambda: left > right,
                "<=": lambda: left <= right,
                ">=": lambda: left >= right,
                "&&": lambda: left and right,
                "||": lambda: left or right,
            }[op]()
        except (KeyError, TypeError, ZeroDivisionError, ValueError):
            return None

    # Types.

    def resolve(self, node):
        return self.resolver.resolve(node)

    def struct_fields(self, name):
        return self.struct_types[name]


def scalar_of_suffix(text: str) -> Scalar | None:
    return {**_INT_SUFFIX, **_FLOAT_SUFFIX}.get(text[-1:]) if text else None

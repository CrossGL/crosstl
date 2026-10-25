"""Symbolic resource and interface indices in CrossGL attributes.

Both source languages that CrossGL imports index resources and interface slots with constant expressions, not only
with literals:

* MSL attribute arguments such as ``[[buffer(index)]]``, ``[[texture(index)]]``, ``[[attribute(index)]]``,
  ``[[id(index)]]`` and ``[[color(index)]]`` are integral constant expressions (MSL 3.2 §5.2.1, §2.13), and Apple's
  samples name them with enumerators of a header shared with the host: ``[[buffer(AAPLBufferIndexVertices)]]``.
* GLSL layout qualifier values are constant integral expressions (GLSL 4.60 §4.4, "layout-qualifier-id =
  layout-qualifier-value", which since GLSL 4.40 may be any constant integral expression), so
  ``layout(location = BASE_LOCATION + 1)`` is valid.

The importers keep those expressions, which carry the binding the host relies on. The generators read an index
attribute as an integer literal, and given anything else they disagreed: GLSL and SPIR-V refused or printed the AST,
the Metal generator dropped the attribute (the compiler then assigns the slot), and the HLSL generator assigned the
next free register or turned a location into a ``location`` semantic. Each of the last three moves the binding
without a diagnostic.

:func:`fold_binding_constants` evaluates, for every target, each index argument that refers to a name: an
enumerator (bare or as ``Enum::Variant``) or an integer ``const`` of the program, combined with C's integer
operators. The argument becomes the integer literal. An index that refers to anything else (a specialization
constant, a name the program does not define, a non-integer value or a negative result) is refused with
:class:`BindingIndexUnresolvedError`, because no target can bind it as written. Arguments that are already literals,
and the ``space1`` and ``set1`` register spellings, are left to the generators.

``[[function_constant(index)]]`` on a program-scope constant declares a Metal function constant with that index (MSL
3.2 §5.8.1), and the samples name the index with an enumerator as well. The same attribute on an argument or a
structure member names a Boolean function constant that makes it conditional, so only the declaration form is
folded; ``constant_id``, the Vulkan spelling, is treated the same way.

WGSL output is the one exception for a name the program does not declare. WGSL has no preprocessor, and the
``@group`` and ``@binding`` attributes take any const-expression (WGSL, "Resource Interface"), so a host that numbers
its bind groups declares those constants in the module text it compiles. Such an index is kept as written, and the
runtime package reports it as a symbolic binding that reflection cannot resolve.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .storage_blocks import _walk

__all__ = [
    "BindingIndexUnresolvedError",
    "HOST_DECLARED_INDEX_TARGETS",
    "INDEX_ATTRIBUTES",
    "SPECIALIZATION_INDEX_ATTRIBUTES",
    "fold_binding_constants",
]

# Attributes whose argument is an index only on a constant's declaration; elsewhere it names a constant.
SPECIALIZATION_INDEX_ATTRIBUTES = frozenset({"constant_id", "function_constant"})

# Targets whose output keeps an index that names an undeclared constant, for the host to declare.
HOST_DECLARED_INDEX_TARGETS = frozenset({"wgsl"})

INDEX_ATTRIBUTES = frozenset(
    {
        "attribute",
        "binding",
        "buffer",
        "color",
        "component",
        "id",
        "index",
        "input_attachment_index",
        "location",
        "raster_order_group",
        "sampler",
        "set",
        "space",
        "texture",
        "threadgroup",
    }
)
INTEGER_TYPES = frozenset(
    {
        "int",
        "uint",
        "short",
        "ushort",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "char",
        "uchar",
        "long",
        "ulong",
        "size_t",
        # CrossGL spellings of the fixed-width integers.
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int64",
        "uint64",
    }
)
REGISTER_SPELLING = re.compile(r"(?i)(?:set|space|[btsu])?\d+")
INTEGER_LITERAL = re.compile(r"(?i)(0x[0-9a-f]+|0b[01]+|0[0-7]*|[1-9][0-9]*)[ul]*")
_AMBIGUOUS = object()


class BindingIndexUnresolvedError(TranslationDiagnostic):
    """An index attribute whose argument is not a constant integer the translator can evaluate."""

    def __init__(self, target, attribute, argument, reason):
        self.missing_capabilities = (f"{target}.symbolic-binding-index",)
        super().__init__(
            f"@{attribute}({argument}) cannot be bound: {reason}",
            target=target,
            code=f"{CODE_PREFIX}binding-index-unresolved",
        )


class _Unresolved(Exception):
    def __init__(self, reason, *, undeclared=False):
        super().__init__(reason)
        self.reason = reason
        self.undeclared = undeclared


def _type_name(node):
    name = getattr(node, "name", None)
    if isinstance(name, str):
        return name
    return node if isinstance(node, str) else None


def _literal_int(value):
    if isinstance(value, bool):
        raise _Unresolved("a boolean is not an index")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    match = INTEGER_LITERAL.fullmatch(text)
    if match is None:
        raise _Unresolved(f"{text} is not an integer")
    digits = match.group(1).lower()
    if digits.startswith(("0x", "0b")):
        return int(digits, 0)
    if len(digits) > 1 and digits.startswith("0"):
        return int(digits, 8)
    return int(digits)


def _c_divide(left, right, operator):
    if right == 0:
        raise _Unresolved("the index divides by zero")
    quotient = abs(left) // abs(right)
    if (left < 0) != (right < 0):
        quotient = -quotient
    return quotient if operator == "/" else left - right * quotient


def _binary(operator, left, right):
    if operator in {"/", "%"}:
        return _c_divide(left, right, operator)
    if operator in {"<<", ">>"}:
        if right < 0:
            raise _Unresolved("the index shifts by a negative amount")
        return left << right if operator == "<<" else left >> right
    operations = {
        "+": lambda: left + right,
        "-": lambda: left - right,
        "*": lambda: left * right,
        "&": lambda: left & right,
        "|": lambda: left | right,
        "^": lambda: left ^ right,
    }
    if operator not in operations:
        raise _Unresolved(f"operator {operator} is not an integer operator")
    return operations[operator]()


class _ConstantTable:
    """Enumerators and integer ``const`` declarations of a program, evaluated on demand."""

    def __init__(self, program):
        self.entries = {}
        self.values = {}
        self.pending = set()
        for node in _walk(program):
            if isinstance(node, A.EnumNode):
                self._add_enum(node)
        for constant in getattr(program, "constants", None) or []:
            self._add_constant(constant)
        for variable in getattr(program, "global_variables", None) or []:
            self._add_constant(variable)

    def _add(self, name, entry):
        if not name:
            return
        if name in self.entries and self.entries[name] is not entry:
            self.entries[name] = _AMBIGUOUS
        else:
            self.entries[name] = entry

    def _add_enum(self, enum):
        previous = None
        for variant in getattr(enum, "variants", None) or []:
            name = getattr(variant, "name", None)
            entry = ("enum", enum, variant, previous)
            self._add(name, entry)
            if getattr(enum, "name", None):
                self._add(f"{enum.name}::{name}", entry)
            previous = entry

    def _add_constant(self, node):
        if isinstance(node, A.ConstantNode):
            declared_type = getattr(node, "const_type", None)
            value = getattr(node, "value", None)
        elif isinstance(node, A.VariableNode):
            qualifiers = {str(q) for q in getattr(node, "qualifiers", None) or []}
            if "const" not in qualifiers and "constant" not in qualifiers:
                return
            declared_type = getattr(node, "var_type", None) or getattr(
                node, "vtype", None
            )
            value = getattr(node, "initial_value", None) or getattr(node, "value", None)
        else:
            return
        name = getattr(node, "name", None)
        if getattr(node, "attributes", None):
            # Specialization and function constants: the host may override the default.
            self._add(name, ("unresolved", f"{name} is a specialization constant"))
        elif _type_name(declared_type) not in INTEGER_TYPES or value is None:
            self._add(name, ("unresolved", f"{name} is not an integer constant"))
        else:
            self._add(name, ("const", node, value))

    def lookup(self, name):
        if name in self.values:
            return self.values[name]
        entry = self.entries.get(name)
        if entry is None:
            raise _Unresolved(
                f"{name} is not an enumerator or integer constant", undeclared=True
            )
        if entry is _AMBIGUOUS:
            raise _Unresolved(f"{name} names several constants")
        if entry[0] == "unresolved":
            raise _Unresolved(entry[1])
        key = id(entry)
        if key in self.pending:
            raise _Unresolved(f"{name} is defined in terms of itself")
        self.pending.add(key)
        try:
            value = self._evaluate_entry(entry)
        finally:
            self.pending.discard(key)
        self.values[name] = value
        return value

    def _evaluate_entry(self, entry):
        if entry[0] == "const":
            return self.evaluate(entry[2])
        _, enum, variant, previous = entry
        if getattr(variant, "value", None) is not None:
            return self.evaluate(variant.value)
        if previous is None:
            return 0
        return self.lookup(f"{enum.name}::{previous[2].name}") + 1

    def evaluate(self, node):
        if isinstance(node, A.LiteralNode):
            return _literal_int(node.value)
        if isinstance(node, A.IdentifierNode):
            return self.lookup(node.name)
        if isinstance(node, A.MemberAccessNode):
            owner = getattr(getattr(node, "object_expr", None), "name", None)
            member = getattr(node, "member", None)
            if isinstance(owner, str) and isinstance(member, str):
                return self.lookup(f"{owner}::{member}")
            raise _Unresolved("a member access is not a constant")
        if isinstance(node, A.BinaryOpNode):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            return _binary(str(node.op), left, right)
        if isinstance(node, A.UnaryOpNode) and not getattr(node, "is_postfix", False):
            operand = self.evaluate(node.operand)
            if node.op == "-":
                return -operand
            if node.op == "+":
                return operand
            raise _Unresolved(f"operator {node.op} is not supported in an index")
        if isinstance(node, A.CastNode):
            if _type_name(node.target_type) not in INTEGER_TYPES:
                raise _Unresolved("the index is cast to a non-integer type")
            return self.evaluate(node.expression)
        if isinstance(node, A.FunctionCallNode):
            callee = _type_name(getattr(node, "name", None)) or _type_name(
                getattr(node, "function", None)
            )
            arguments = list(getattr(node, "arguments", None) or [])
            if callee in INTEGER_TYPES and len(arguments) == 1:
                return self.evaluate(arguments[0])
            raise _Unresolved("a function call is not a constant")
        raise _Unresolved(f"{type(node).__name__} is not a constant")


def _refers_to_names(node):
    """True when ``node`` names a declaration rather than spelling a literal or a register."""
    if isinstance(node, A.LiteralNode):
        return False
    if isinstance(node, A.IdentifierNode):
        return REGISTER_SPELLING.fullmatch(str(node.name)) is None
    if isinstance(node, A.UnaryOpNode):
        return _refers_to_names(node.operand)
    if isinstance(node, A.ASTNode):
        return True
    return False


def _spelling(node):
    if isinstance(node, A.IdentifierNode):
        return node.name
    if isinstance(node, A.LiteralNode):
        return str(node.value)
    return type(node).__name__


def _fold_attribute(attribute, table, target):
    arguments = getattr(attribute, "arguments", None)
    if not isinstance(arguments, list):
        return 0
    changed = 0
    for index, argument in enumerate(arguments):
        if not _refers_to_names(argument):
            continue
        try:
            value = table.evaluate(argument)
        except _Unresolved as error:
            if error.undeclared and target in HOST_DECLARED_INDEX_TARGETS:
                continue
            reason = error.reason
            if not isinstance(argument, A.IdentifierNode):
                reason = f"the expression is not a constant integer: {reason}"
            raise BindingIndexUnresolvedError(
                target, attribute.name, _spelling(argument), reason
            ) from None
        if value < 0:
            raise BindingIndexUnresolvedError(
                target,
                attribute.name,
                _spelling(argument),
                f"it evaluates to {value}, and an index is non-negative",
            )
        arguments[index] = A.LiteralNode(value, A.PrimitiveType("int"))
        changed += 1
    return changed


def fold_binding_constants(program, target) -> int:
    """Replace the symbolic index arguments of ``program``'s attributes by integers; returns how many."""
    if not isinstance(program, A.ShaderNode):
        return 0
    target = str(target or "")
    table = None
    changed = 0
    declarations = {
        id(node)
        for node in (getattr(program, "constants", None) or []) + (
            getattr(program, "global_variables", None) or []
        )
    }
    for node in _walk(program):
        for attribute in getattr(node, "attributes", None) or []:
            if not isinstance(attribute, A.AttributeNode):
                continue
            name = str(getattr(attribute, "name", "")).lower()
            if name in SPECIALIZATION_INDEX_ATTRIBUTES:
                if id(node) not in declarations:
                    continue
            elif name not in INDEX_ATTRIBUTES:
                continue
            if not any(_refers_to_names(a) for a in attribute.arguments or []):
                continue
            if table is None:
                table = _ConstantTable(program)
            changed += _fold_attribute(attribute, table, target)
    return changed

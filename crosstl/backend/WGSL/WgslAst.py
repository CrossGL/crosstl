"""The syntax tree of a WGSL module, as :mod:`.WgslParser` builds it (W3C WGSL, section 16 grammar).

Types are written as expressions: ``vec3<f32>``, ``array<f32, 4>`` and ``ptr<function, i32>`` are identifiers with
template arguments (:class:`Ident`), as the grammar writes a ``template_elaborated_ident``. :mod:`.wgsl_types`
resolves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Alias",
    "Assign",
    "Attribute",
    "Binary",
    "Block",
    "Break",
    "BreakIf",
    "Call",
    "CallStatement",
    "Const",
    "ConstAssert",
    "Continue",
    "Directive",
    "Discard",
    "For",
    "Function",
    "GlobalVar",
    "Ident",
    "If",
    "Increment",
    "Index",
    "Literal",
    "Local",
    "Loop",
    "Member",
    "Module",
    "Override",
    "Param",
    "Return",
    "Struct",
    "StructMember",
    "Switch",
    "SwitchClause",
    "Unary",
    "While",
]


@dataclass
class Attribute:
    name: str
    args: list = field(default_factory=list)
    line: int = 0


# Expressions.


@dataclass
class Literal:
    """A numeric or boolean literal; ``kind`` is ``int``, ``float`` or ``bool`` and ``text`` its spelling."""

    kind: str
    text: str
    line: int = 0


@dataclass
class Ident:
    """A name, with the template arguments of a ``template_elaborated_ident`` (``None`` when it has none)."""

    name: str
    template_args: list | None = None
    line: int = 0


@dataclass
class Call:
    callee: Ident
    args: list
    line: int = 0


@dataclass
class Member:
    base: object
    name: str
    line: int = 0


@dataclass
class Index:
    base: object
    index: object
    line: int = 0


@dataclass
class Unary:
    """``-e``, ``!e``, ``~e``, ``*e`` (indirection) or ``&e`` (address-of)."""

    op: str
    operand: object
    line: int = 0


@dataclass
class Binary:
    op: str
    left: object
    right: object
    line: int = 0


# Statements.


@dataclass
class Block:
    statements: list
    line: int = 0


@dataclass
class Local:
    """A ``var``, ``let`` or ``const`` declaration in a function body (``kind``)."""

    kind: str
    name: str
    type: object | None
    init: object | None
    template_args: list | None = None
    line: int = 0


@dataclass
class Assign:
    """``lhs op rhs``; ``op`` is ``=`` or a compound operator, and ``lhs`` is ``None`` for ``_ = rhs``."""

    lhs: object | None
    op: str
    rhs: object
    line: int = 0


@dataclass
class Increment:
    target: object
    op: str
    line: int = 0


@dataclass
class CallStatement:
    call: Call
    line: int = 0


@dataclass
class If:
    condition: object
    then: Block
    otherwise: object | None = None
    line: int = 0


@dataclass
class SwitchClause:
    """``selectors`` holds expressions and the string ``default``."""

    selectors: list
    body: Block
    line: int = 0


@dataclass
class Switch:
    selector: object
    clauses: list
    line: int = 0


@dataclass
class BreakIf:
    condition: object
    line: int = 0


@dataclass
class Loop:
    body: Block
    continuing: Block | None = None
    break_if: BreakIf | None = None
    line: int = 0


@dataclass
class For:
    init: object | None
    condition: object | None
    update: object | None
    body: Block
    line: int = 0


@dataclass
class While:
    condition: object
    body: Block
    line: int = 0


@dataclass
class Break:
    line: int = 0


@dataclass
class Continue:
    line: int = 0


@dataclass
class Return:
    value: object | None = None
    line: int = 0


@dataclass
class Discard:
    line: int = 0


@dataclass
class ConstAssert:
    condition: object
    line: int = 0


# Module-scope declarations.


@dataclass
class Directive:
    """``enable``, ``requires`` or ``diagnostic``, with its names."""

    kind: str
    names: list
    line: int = 0


@dataclass
class StructMember:
    name: str
    type: object
    attributes: list = field(default_factory=list)
    line: int = 0


@dataclass
class Struct:
    name: str
    members: list
    line: int = 0


@dataclass
class Alias:
    name: str
    type: object
    line: int = 0


@dataclass
class GlobalVar:
    name: str
    type: object | None
    init: object | None
    template_args: list | None = None
    attributes: list = field(default_factory=list)
    line: int = 0


@dataclass
class Const:
    name: str
    type: object | None
    init: object
    line: int = 0


@dataclass
class Override:
    name: str
    type: object | None
    init: object | None
    attributes: list = field(default_factory=list)
    line: int = 0


@dataclass
class Param:
    name: str
    type: object
    attributes: list = field(default_factory=list)
    line: int = 0


@dataclass
class Function:
    name: str
    params: list
    return_type: object | None
    body: Block
    attributes: list = field(default_factory=list)
    return_attributes: list = field(default_factory=list)
    line: int = 0


@dataclass
class Module:
    directives: list
    declarations: list

"""Same-width 32-bit word views of storage inside resource handles.

A view whose element is one 32-bit scalar over storage whose element is one
32-bit scalar, or a record whose only member is one, addresses whole backing
words (see ``storage_word_views``). Resource-aggregate lowering represents
such a pointer as a handle of the view element whose identity still names the
backing storage and whose offset counts view elements, which are also backing
elements. Unlike narrowing views, a word view keeps its source's write access:
a store replaces one whole word, so it never races with a neighbour's store.

Loads and stores are bit casts of the backing word. Integer atomics through
a word view need the backing word's signedness. Float atomics are a
compare-and-swap loop on ``uint`` backing words; ``atomicLoad``,
``atomicStore`` and ``atomicExchange`` are the integer atomics on the bits.
"""

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    FunctionCallNode,
    IdentifierNode,
    PrimitiveType,
    ReturnNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .pointer_reinterpret import scalar_storage_layout
from .storage_word_views import layout_scalar_kind, reinterpret_source_layout

_BIT_CASTS = {"float": "asfloat", "int": "asint", "uint": "asuint"}


class WordViewAtomicError(ValueError):
    """Raised for an atomic the backing words cannot express."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def word_view(source_element, target_element, structs_by_name):
    """Return ``(source_layout, target_layout, member)`` for a word view."""

    if source_element == target_element:
        return None
    source, member = reinterpret_source_layout(source_element, structs_by_name)
    target = scalar_storage_layout(target_element)
    if (
        source is None
        or target is None
        or source.bit_width != 32
        or target.bit_width != 32
    ):
        return None
    return source, target, member


def _call(name, arguments):
    return FunctionCallNode(IdentifierNode(name), arguments)


def bit_cast(value, source_layout, target_layout):
    """Reinterpret the 32 bits of ``value`` from one layout to another."""

    source_kind = layout_scalar_kind(source_layout)
    target_kind = layout_scalar_kind(target_layout)
    if source_kind == target_kind:
        return value
    return _call(_BIT_CASTS[target_kind], [value])


def word_load(word, view):
    source, target, _member = view
    return bit_cast(word, source, target)


def word_store(word, value, operator, view):
    """Statements storing ``value`` with ``operator``; they return the result."""

    source, target, _member = view
    stored = value
    if operator != "=":
        stored = BinaryOpNode(bit_cast(word, source, target), operator[:-1], value)
        stored = _call(layout_scalar_kind(target), [stored])
    return [
        AssignmentNode(deepcopy(word), bit_cast(stored, target, source)),
        ReturnNode(bit_cast(deepcopy(word), source, target)),
    ]


def word_atomic(operation, word, values, view):
    """Statements performing ``operation`` on one backing word.

    ``values`` are the helper's value parameters as identifiers. The
    statements return what the source atomic returns.
    """

    source, target, _member = view
    physical = layout_scalar_kind(source)
    logical = layout_scalar_kind(target)
    if logical != "float":
        if logical != physical:
            raise WordViewAtomicError("atomic-signedness-mismatch")
        call = _call(operation, [word, *values])
        if operation == "atomicStore":
            return [call, ReturnNode()]
        return [ReturnNode(call)]
    if physical != "uint":
        raise WordViewAtomicError("float-atomic-backing-word-kind")

    def to_bits(value):
        return bit_cast(value, target, source)

    def to_float(value):
        return bit_cast(value, source, target)

    if operation == "atomicLoad":
        return [ReturnNode(to_float(_call("atomicLoad", [word])))]
    if operation == "atomicStore":
        return [_call("atomicStore", [word, to_bits(values[0])]), ReturnNode()]
    if operation == "atomicExchange":
        exchanged = _call("atomicExchange", [word, to_bits(values[0])])
        return [ReturnNode(to_float(exchanged))]
    if operation != "atomicAdd":
        raise WordViewAtomicError("unsupported-float-atomic-operation")
    desired = to_bits(
        BinaryOpNode(to_float(IdentifierNode("expected")), "+", values[0])
    )
    exchanged = _call(
        "atomicCompareExchangeWeak",
        [deepcopy(word), IdentifierNode("expected"), desired],
    )
    return [
        VariableNode("expected", PrimitiveType("uint"), _call("atomicLoad", [word])),
        # A failed exchange stores the observed word into ``expected``.
        WhileNode(UnaryOpNode("!", exchanged), BlockNode([])),
        ReturnNode(to_float(IdentifierNode("expected"))),
    ]

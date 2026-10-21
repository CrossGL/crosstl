"""Narrowing integer views of storage buffers inside resource handles.

C++-style sources reinterpret a word buffer as bytes (``(const device
uint8_t*)words``) and keep the byte pointer in an aggregate. Resource-aggregate
lowering represents such a pointer as a handle of the view's element whose
identity still names the word buffer and whose offset counts view elements. A
load through the handle then extracts one view element from the word that holds
it. GPU storage is little-endian on every supported target, so element ``i`` of
a view with ``ratio`` elements per word is bits ``(i % ratio) * width`` upward
of word ``i / ratio``.

Only reads through unsigned views of integer storage are representable this
way. A narrower store would need a read-modify-write of the shared word, which
races with the neighbours' stores, so views never grant write access.
"""

from copy import deepcopy
from dataclasses import dataclass

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
)
from .pointer_reinterpret import scalar_storage_layout


@dataclass(frozen=True)
class StorageView:
    """How many ``target`` elements one ``source`` storage element holds."""

    source: str
    target: str
    ratio: int
    width: int


def storage_view(source_element, target_element):
    """The narrowing view of ``source_element`` storage, or None.

    The source must be integer storage, the target an unsigned integer whose
    width divides the source's. Wider, signed and floating views are left to
    the caller, which refuses them.
    """
    source = scalar_storage_layout(source_element)
    target = scalar_storage_layout(target_element)
    if source is None or target is None:
        return None
    if source.kind != "integer" or target.kind != "integer" or target.signed:
        return None
    if target.bit_width >= source.bit_width or source.bit_width % target.bit_width:
        return None
    return StorageView(
        source_element,
        target_element,
        source.bit_width // target.bit_width,
        target.bit_width,
    )


def _integer(value):
    return LiteralNode(value, PrimitiveType("int"))


def _call(name, arguments):
    return FunctionCallNode(IdentifierNode(name), arguments)


def _position(value):
    # Positions are int64_t; typed operands keep every target's overloads exact.
    return _call("int64_t", [_integer(value)])


def view_load(view, storage, index):
    """Read view element ``index`` from ``storage``, an array of source words.

    Positions in a view are never negative, and widths are powers of two, so
    the lane within the word is a mask of ``index``. ``index`` is evaluated
    twice, so callers pass a side-effect-free expression, as the helper's
    ``reference.offset + index`` is.
    """
    word = ArrayAccessNode(storage, BinaryOpNode(index, "/", _position(view.ratio)))
    lane = _call(
        "uint", [BinaryOpNode(deepcopy(index), "&", _position(view.ratio - 1))]
    )
    shift = BinaryOpNode(lane, "*", _call("uint", [_integer(view.width)]))
    mask = _call(view.source, [_integer((1 << view.width) - 1)])
    return _call(
        view.target,
        [BinaryOpNode(BinaryOpNode(word, ">>", shift), "&", mask)],
    )

"""Which CrossGL constant globals the Rust generator can write as ``const`` items.

A ``#![no_std]`` rust-gpu crate has no lazily initialized statics, so a CrossGL ``const`` global is a Rust ``const``
item, whose initializer Rust evaluates at compile time (The Rust Reference, "Constant evaluation"). That holds for
literals, other constants, primitive arithmetic and casts, array and struct literals, and calls of ``const fn``s. The
generator's own spellings of vector, matrix and struct construction use only these glam and generated constructors,
which are ``const fn`` in glam 0.29 and later, and the generated ``new`` of every struct:
``new``, ``splat``, ``from_cols`` and ``from_cols_array``. Operators on glam vectors and matrices are trait methods,
which are not ``const``, as are math built-ins and user functions. An initializer that uses any of them is refused
instead of being written as a ``const`` Rust would reject.
"""

from __future__ import annotations

import re

from ..ast import BinaryOpNode, LiteralNode, UnaryOpNode
from .ast_walk import walk_ast_values
from .scalar_constant_folding import ScalarConstant, ScalarConstantFolder

CONST_CONSTRUCTORS = frozenset({"new", "splat", "from_cols", "from_cols_array"})
_CALL = re.compile(r"(::|\.)?\s*\b(r#)?([A-Za-z_]\w*)\s*\(")
_KEYWORDS = frozenset({"if", "while", "match", "return"})


def _text_is_const(initializer: str) -> bool:
    for match in _CALL.finditer(initializer):
        separator, _raw, name = match.groups()
        if separator == "::":
            if name not in CONST_CONSTRUCTORS:
                return False
        elif separator == ".":
            return False
        elif name not in _KEYWORDS:
            return False
    return True


def _operators_are_scalar(generator, value) -> bool:
    for node in walk_ast_values(value):
        if isinstance(node, (BinaryOpNode, UnaryOpNode)):
            result_type = generator.expression_result_type(node)
            if generator.normalize_scalar_type(result_type) is None:
                return False
    return True


def rust_const_initializer_reason(generator, value, initializer) -> str | None:
    """Why ``initializer`` (generated from CrossGL ``value``) cannot be a Rust ``const`` initializer, or ``None``."""
    if value is None:
        return None
    if not _operators_are_scalar(generator, value):
        return "its initializer applies operators to vectors or matrices, which Rust does not evaluate in a const"
    if not _text_is_const(str(initializer)):
        return "its initializer calls a function that is not a const fn"
    return None


_INTEGER_KINDS = {"int": "int", "uint": "uint", "i32": "int", "u32": "uint"}


def _integer_kind(type_value) -> str | None:
    name = getattr(type_value, "name", type_value)
    return _INTEGER_KINDS.get(str(name)) if name is not None else None


def rust_integer_constants(ast) -> dict[str, ScalarConstant]:
    """The ``int``/``uint`` constant globals of ``ast`` whose initializers fold, in declaration order."""
    values: dict[str, ScalarConstant] = {}
    folder = ScalarConstantFolder(values.get, _integer_kind)
    for node in getattr(ast, "constants", None) or []:
        name = getattr(node, "name", None)
        kind = _integer_kind(getattr(node, "const_type", getattr(node, "vtype", None)))
        if not name or kind is None:
            continue
        folded = folder.fold(getattr(node, "value", None))
        if folded is not None and isinstance(folded.value, int) and not isinstance(folded.value, bool):
            values[name] = folded
    return values


def fold_array_size(generator, size):
    """An array extent as an ``int``: a literal, or an expression over integer constants; ``None`` otherwise.

    Rust array lengths are ``usize`` constants, so an extent such as ``N * 2`` is folded with the program's integer
    constants. An extent that does not fold, or folds to zero or less, stays unsized and is refused downstream.
    """
    if size is None:
        return None
    if isinstance(size, LiteralNode):
        return size.value
    if isinstance(size, (int, str)):
        return size
    values = getattr(generator, "rust_integer_constant_values", None) or {}
    folded = ScalarConstantFolder(values.get, _integer_kind).fold(size)
    if folded is None or not isinstance(folded.value, int) or folded.value <= 0:
        return None
    return folded.value

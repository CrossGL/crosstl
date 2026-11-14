"""Signed ``atomicMin``/``atomicMax`` on a byte-address buffer as a compare-exchange loop on its uint word.

``RWByteAddressBuffer::InterlockedMin``/``InterlockedMax`` compare as signed integers when the value is an ``int``
(DXIL selects ``IMin``/``IMax`` from the value's type). GLSL and Metal bind the buffer as ``uint`` words, whose
``atomicMin``/``atomicMax`` compare as unsigned, so the word lowering performs the signed operation with a
compare-exchange loop on the word, which every target with buffer atomics provides::

    uint word = <address> >> 2u;
    int value = <value>;
    uint expected = uint(value);
    uint seen = atomicCompareExchange(buffer[word], expected, uint(min(int(expected), value)));
    while (seen != expected) {
        expected = seen;
        seen = atomicCompareExchange(buffer[word], expected, uint(min(int(expected), value)));
    }

Every access to the word is atomic. The first guess is the value itself, for which a successful exchange stores the
word unchanged. The loop ends once an exchange found the word it compared with, so ``expected`` is then the original
word, which converts into the receiving variable as the signed overload's ``out int original_value`` does. Each failed exchange means
another invocation changed the word, so the loop is lock-free.

The loop needs a statement of its own: the atomic as a statement, the right-hand side of a plain assignment
statement, or the initializer of a declaration directly inside a block (the loop's temporaries are then declared in
that block, before the declaration). Elsewhere the word lowering refuses the atomic.
"""

from __future__ import annotations

from .. import ast as A
from .rewrite import is_pure

__all__ = ["signed_ordered_atomic_replacement"]

_ORDERED = {"atomicMin": "min", "atomicMax": "max"}


def _call(name, arguments):
    return A.FunctionCallNode(A.IdentifierNode(name), list(arguments))


def _call_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, A.IdentifierNode):
        return function.name
    return function if isinstance(function, str) else None


def _signed_ordered_call(lowerer, call):
    """``(name, args)`` when ``call`` is a signed atomicMin/atomicMax on a byte-address buffer, else ``None``."""
    if not isinstance(call, A.FunctionCallNode) or id(call) in lowerer.generated:
        return None
    name = _call_name(call)
    args = list(call.arguments or [])
    if name not in _ORDERED or len(args) != 3:
        return None
    if lowerer.buffer_kind(args[0]) is None or lowerer.scalar_kind(args[2]) != "i":
        return None
    return name, args


def _variable(name, type_name, value):
    return A.VariableNode(name, A.PrimitiveType(type_name), value)


def _exchange(lowerer, buffer, word, expected, value, operation):
    """``seen = atomicCompareExchange(buffer[word], expected, uint(op(int(expected), value)))`` as an expression."""
    desired = _call(
        "uint",
        [
            _call(
                operation,
                [_call("int", [A.IdentifierNode(expected)]), A.IdentifierNode(value)],
            )
        ],
    )
    target = lowerer.element(buffer, A.IdentifierNode(word))
    return _call("atomicCompareExchange", [target, A.IdentifierNode(expected), desired])


def _loop(lowerer, name, args):
    """The statements of the loop and the name of the variable that holds the original word."""
    buffer, address, value = args
    word_name = lowerer.fresh_name("crosstl_byte_address_word")
    value_name = lowerer.fresh_name("crosstl_byte_address_value")
    expected_name = lowerer.fresh_name("crosstl_byte_address_expected")
    seen_name = lowerer.fresh_name("crosstl_byte_address_seen")
    operation = _ORDERED[name]

    def exchange():
        return _exchange(
            lowerer, buffer, word_name, expected_name, value_name, operation
        )

    retry = A.BlockNode(
        [
            A.ExpressionStatementNode(
                A.AssignmentNode(
                    A.IdentifierNode(expected_name), A.IdentifierNode(seen_name)
                )
            ),
            A.ExpressionStatementNode(
                A.AssignmentNode(A.IdentifierNode(seen_name), exchange())
            ),
        ]
    )
    statements = [
        _variable(word_name, "uint", lowerer.word_index(address)),
        _variable(value_name, "int", value),
        _variable(expected_name, "uint", _call("uint", [A.IdentifierNode(value_name)])),
        _variable(seen_name, "uint", exchange()),
        A.WhileNode(
            A.BinaryOpNode(
                A.IdentifierNode(seen_name), "!=", A.IdentifierNode(expected_name)
            ),
            retry,
        ),
    ]
    return statements, expected_name


def _original(lowerer, expected_name, receiver):
    """The original word converted into ``receiver`` (a variable or a declared type).

    The signed overload's ``original_value`` is an ``int``, which converts into the receiver as HLSL converts an
    ``out int`` argument: a ``uint`` receiver keeps its bits and a ``float`` receiver gets its numeric value.
    """
    original = A.IdentifierNode(expected_name)
    kind = lowerer.scalar_kind(receiver)
    if kind == "i":
        return _call("int", [original])
    if kind == "f":
        return _call("float", [_call("int", [original])])
    return original


def signed_ordered_atomic_replacement(lowerer, node, in_block):
    """The lowering of a statement that is, assigns or declares a signed byte-address atomicMin/atomicMax.

    Returns ``None`` when ``node`` is no such statement, ``False`` after a refusal, a ``BlockNode`` for an atomic
    or an assignment statement, and a list of statements, to be spliced into the enclosing block in place of
    ``node``, for a declaration (``in_block`` tells whether ``node`` is an item of a block's statements).
    """
    receiver = None
    declaration = None
    if isinstance(node, A.ExpressionStatementNode):
        call = node.expression
        if isinstance(call, A.AssignmentNode) and call.operator == "=":
            receiver, call = call.target, call.value
    elif isinstance(node, A.VariableNode) and in_block:
        declaration, call = node, node.initial_value
    else:
        return None
    signed = _signed_ordered_call(lowerer, call)
    if signed is None:
        return None
    name, args = signed
    if not is_pure(args[0]):
        lowerer.refuse(
            f"{name} on a byte-address buffer whose index has side effects: the signed form is a "
            "compare-exchange loop that evaluates the buffer on every attempt"
        )
        return False
    statements, expected_name = _loop(lowerer, name, args)
    if declaration is not None:
        declaration.initial_value = _original(
            lowerer, expected_name, declaration.var_type
        )
        return [*statements, declaration]
    if receiver is not None:
        statements.append(
            A.ExpressionStatementNode(
                A.AssignmentNode(receiver, _original(lowerer, expected_name, receiver))
            )
        )
    return A.BlockNode(statements)

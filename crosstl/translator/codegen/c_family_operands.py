"""Operand parenthesization for C-family emitters (Slang, HLSL, MSL, GLSL).

Emitters build expressions as text. A rendered operand placed before a postfix
operator (``.member``, ``[index]``, ``->member``, ``++``) or after a prefix unary
operator must be parenthesized unless the text already binds at least as
tightly as that operator. Otherwise ``(p + q).x`` is emitted as ``p + q.x`` and
``-(-a)`` as the pre-decrement ``--a``.

The checks work on the rendered text rather than on AST node types because
lowerings can turn any node into arbitrary text. They are conservative: text
that is not recognised as a primary or postfix chain is parenthesized, which
never changes the meaning of a well-formed expression.
"""

_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_CLOSERS = {"(": ")", "[": "]"}
_GROUP_ENDS = frozenset(")]")
_PREFIX_OPERATOR_CHARS = frozenset("-+!~*&")
# A prefix operator ending in one of these characters merges with an operand
# that starts with the same character: ``-`` ``-a`` lexes as ``--a``.
_MERGING_CHARS = frozenset("-+&")


def _is_numeric_literal(text):
    return text[0].isdigit() or (text[0] == "." and text[1:2].isdigit())


def _group_end(text):
    """The index just past the group that opens ``text``."""
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def is_postfix_chain(text):
    """Whether ``text`` is one primary expression followed only by postfix parts.

    Accepted: identifiers, literals, ``a::b``, calls, index and member accesses,
    ``->`` accesses, and one parenthesized group, in any chain such as
    ``f(x)[i].y`` or ``(a + b).x``. Anything with a top-level operator, space,
    comma, ``?`` or template angle bracket is rejected.
    """
    text = text.strip()
    if not text:
        return False
    stack = []
    previous = ""
    leading_group_end = None
    index = 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            return False
        if stack:
            if char in _CLOSERS:
                stack.append(_CLOSERS[char])
            elif char == stack[-1]:
                stack.pop()
                if not stack:
                    previous = char
            elif char in ")]":
                return False
            index += 1
            continue
        if char in _CLOSERS:
            # A group opens a chain, or follows a name or another group as a
            # call or an index. ``(float)x`` fails below, at ``x``, and
            # ``(T)(x)`` here, because it may be a cast rather than a call.
            if previous and previous not in _IDENTIFIER_CHARS:
                if previous not in _GROUP_ENDS:
                    return False
                if char == "(" and leading_group_end == index:
                    return False
            if index == 0:
                leading_group_end = _group_end(text)
            stack.append(_CLOSERS[char])
        elif char in _IDENTIFIER_CHARS:
            if previous in _GROUP_ENDS:
                return False
            previous = char
        elif char == ".":
            if not previous:
                if not text[1:2].isdigit():
                    return False
            previous = char
        elif text.startswith("::", index) or text.startswith("->", index):
            if previous not in _IDENTIFIER_CHARS and previous not in _GROUP_ENDS:
                return False
            index += 2
            previous = "."
            continue
        else:
            return False
        index += 1
    return not stack and previous not in {".", ""}


def postfix_operand(text):
    """``text`` ready to take ``.member``, ``[index]``, ``->member`` or ``++``."""
    text = str(text)
    stripped = text.strip()
    if not stripped:
        return text
    if is_postfix_chain(stripped) and not _is_numeric_literal(stripped):
        return text
    return f"({stripped})"


def _prefix_operand_shape(text):
    """``"chain"``, ``"prefixed"`` (prefix operators then a chain) or ``None``."""
    if is_postfix_chain(text):
        return "chain"
    body = text.lstrip("".join(_PREFIX_OPERATOR_CHARS))
    if body != text and is_postfix_chain(body):
        return "prefixed"
    return None


def prefix_unary_operand(operator, text):
    """``text`` ready to follow the prefix unary ``operator``.

    A chain needs no parentheses. A prefixed operand keeps none unless its first
    character would merge with the operator into another token (``-`` ``-a``,
    ``+`` ``+a``, ``&`` ``&a``). Everything else is parenthesized.
    """
    text = str(text)
    stripped = text.strip()
    if not stripped:
        return text
    shape = _prefix_operand_shape(stripped)
    if shape == "chain":
        return text
    if shape == "prefixed":
        operator = str(operator)
        merges = (
            bool(operator)
            and operator[-1] in _MERGING_CHARS
            and operator[-1] == stripped[0]
        )
        if not merges:
            return text
    return f"({stripped})"

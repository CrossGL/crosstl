"""Operator spellings shared by the C-family shading-language emitters.

GLSL, HLSL and the Metal Shading Language spell every CrossGL operator the same way, so one table serves all three.
An expression node carries either the operator itself (``"+"``, which the CrossGL parser stores) or the token name
an importer produced (``"PLUS"``, ``"POST_INCREMENT"``). Both resolve here.

Anything else has no spelling: emitting it verbatim would produce text that either fails to compile or, worse,
parses as a different expression. :func:`refuse_unknown_operator` records the refusal in the generation's
refusal sink, so the translation stops with the emitter's typed refusal instead.
"""

from __future__ import annotations

from .lowering_refusals import refuse_unsupported

__all__ = [
    "C_FAMILY_OPERATORS",
    "OPERATOR_TOKEN_SPELLINGS",
    "c_family_operator_spelling",
    "map_c_family_operator",
    "refuse_unknown_operator",
]

# Operators of GLSL 4.60 section 5.1, HLSL and MSL (whose expressions are those of C++14) that CrossGL expresses with
# the same spelling and meaning. GLSL's ``^^`` has no HLSL or MSL counterpart; importers lower it to ``!=`` on booleans.
C_FAMILY_OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
        "~",
        "!",
        "<<",
        ">>",
        "<",
        ">",
        "<=",
        ">=",
        "==",
        "!=",
        "&&",
        "||",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "<<=",
        ">>=",
        "++",
        "--",
        ",",
    }
)

# Token names the CrossGL lexer and the importers' lexers attach to operator nodes.
OPERATOR_TOKEN_SPELLINGS = {
    "PLUS": "+",
    "MINUS": "-",
    "MULTIPLY": "*",
    "DIVIDE": "/",
    "MOD": "%",
    "BITWISE_AND": "&",
    "BITWISE_OR": "|",
    "BITWISE_XOR": "^",
    "BITWISE_NOT": "~",
    "NOT": "!",
    "BITWISE_SHIFT_LEFT": "<<",
    "BITWISE_SHIFT_RIGHT": ">>",
    "LESS_THAN": "<",
    "GREATER_THAN": ">",
    "LESS_EQUAL": "<=",
    "GREATER_EQUAL": ">=",
    "EQUAL": "==",
    "NOT_EQUAL": "!=",
    "AND": "&&",
    "LOGICAL_AND": "&&",
    "OR": "||",
    "LOGICAL_OR": "||",
    "EQUALS": "=",
    "ASSIGN_ADD": "+=",
    "ASSIGN_SUB": "-=",
    "ASSIGN_MUL": "*=",
    "ASSIGN_DIV": "/=",
    "ASSIGN_MOD": "%=",
    "ASSIGN_AND": "&=",
    "ASSIGN_OR": "|=",
    "ASSIGN_XOR": "^=",
    "ASSIGN_SHIFT_LEFT": "<<=",
    "ASSIGN_SHIFT_RIGHT": ">>=",
    "INCREMENT": "++",
    "DECREMENT": "--",
    # The fixity of these four is also read from the node's operator name or ``is_postfix``; the spelling is shared.
    "PRE_INCREMENT": "++",
    "POST_INCREMENT": "++",
    "PRE_DECREMENT": "--",
    "POST_DECREMENT": "--",
    "COMMA": ",",
}


def c_family_operator_spelling(op):
    """The GLSL/HLSL/MSL spelling of ``op``, or ``None`` when CrossGL defines no such operator."""
    if not isinstance(op, str):
        return None
    if op in C_FAMILY_OPERATORS:
        return op
    return OPERATOR_TOKEN_SPELLINGS.get(op)


def refuse_unknown_operator(op, target_label):
    """Refuse ``op`` for the target named ``target_label`` and return a placeholder that is never emitted.

    Inside a generation the reason goes to the generation's refusal sink, so the program's other refusals are still
    collected and the emitter raises its typed refusal once the program is rendered. Outside one there is no output
    to protect, and the target's typed refusal is raised at once.
    """
    return refuse_unsupported(
        target_label, "operator", f"{op!r} has no {target_label} spelling", str(op)
    )


def map_c_family_operator(op, target_label):
    """``op`` spelled for a C-family target, refusing an operator CrossGL does not define.

    ``None`` and ``""`` stand for a node without an operator: the emitters ask for the operator of arbitrary
    expressions when they classify them, so that question gets the same empty answer back.
    """
    if op is None or op == "":
        return op
    spelling = c_family_operator_spelling(op)
    if spelling is not None:
        return spelling
    return refuse_unknown_operator(op, target_label)

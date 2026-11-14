"""Fold MSL's ``signed``/``unsigned`` integer spellings inside template arguments.

MSL inherits C++'s multi-word integer spellings (MSL 2.1): ``unsigned int`` is ``uint``,
``unsigned short`` is ``ushort``, ``unsigned char`` is ``uchar``, ``unsigned long`` is
``ulong``, a bare ``unsigned`` is ``uint``, and ``signed T`` is ``T``. The type parser already
folds them at the top level of a declaration. Template arguments such as
``texture2d<unsigned int>`` are collected as raw tokens instead, so they are folded here
before the argument list is spelled, which keeps every later texture and vector mapping on the
single canonical name.
"""

SIGNED_PREFIXES = frozenset({"signed", "unsigned"})
INTEGER_BASES = frozenset({"char", "short", "int", "long"})
UNSIGNED_NAMES = {
    "char": "uchar",
    "short": "ushort",
    "int": "uint",
    "long": "ulong",
}


def fold_signed_type_spellings(tokens):
    """Return ``tokens`` with every ``signed``/``unsigned`` spelling folded to one name."""
    folded = []
    index = 0
    count = len(tokens)
    while index < count:
        token = str(tokens[index])
        if token not in SIGNED_PREFIXES:
            folded.append(token)
            index += 1
            continue
        following = str(tokens[index + 1]) if index + 1 < count else ""
        if following in INTEGER_BASES:
            base = following
            index += 2
        else:
            base = "int"
            index += 1
        folded.append(UNSIGNED_NAMES[base] if token == "unsigned" else base)
    return folded


SIZED_INTEGER_NAMES = frozenset({"short", "ushort", "long", "ulong"})


def fold_integer_type_spellings(tokens):
    """Return ``tokens`` folded like :func:`fold_signed_type_spellings`, also dropping the optional ``int`` of
    ``short int``, ``long int``, ``unsigned short int`` and ``unsigned long int``, so that every C++ spelling of
    an integer scalar is one Metal name (``constexpr unsigned int N`` declares a ``uint``).
    """
    folded = []
    for token in fold_signed_type_spellings(tokens):
        if token == "int" and folded and folded[-1] in SIZED_INTEGER_NAMES:
            continue
        folded.append(token)
    return folded

"""The value of a CrossGL character literal on targets without a C character type.

CrossGL's ``char`` is a signed 8-bit integer (see the storage layouts in
pointer_reinterpret.py), and a character literal such as ``'A'`` or ``'\\n'``
denotes the code of its one character, as an integer character constant does
in C (C17 section 6.4.4.4). Rust's ``char`` is a 32-bit Unicode scalar value and
Mojo has no character type, so those targets spell ``char`` as an 8-bit integer
and a character literal as its code.
"""

# C17 section 6.4.4.4: the simple escape sequences and the octal escape \0.
C_SIMPLE_ESCAPES = {
    "'": 39,
    '"': 34,
    "?": 63,
    "\\": 92,
    "a": 7,
    "b": 8,
    "f": 12,
    "n": 10,
    "r": 13,
    "t": 9,
    "v": 11,
    "0": 0,
}


def char_literal_code(text, refusal, target):
    """Return the code of the CrossGL character literal whose body is ``text``.

    ``refusal`` is the target's typed refusal class. A body that is not one
    basic character or one simple escape has no single-byte value: C gives a
    multibyte or multi-character constant an implementation-defined value.
    """
    body = str(text)
    if len(body) == 1 and ord(body) < 128:
        return ord(body)
    if len(body) == 2 and body[0] == "\\" and body[1] in C_SIMPLE_ESCAPES:
        return C_SIMPLE_ESCAPES[body[1]]
    raise refusal(
        f"{target} char literal '{body}': CrossGL char is a signed 8-bit integer, "
        "and only a basic character or a simple escape has a value that fits it "
        "(C17 section 6.4.4.4)"
    )

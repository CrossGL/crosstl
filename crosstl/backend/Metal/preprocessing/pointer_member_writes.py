"""Whether a struct method changes one of the struct's pointer members itself.

Direct pointer-member promotion replaces a pointer member with the constructor argument it was initialized from and
passes that argument to every method by value. A method that advances, reassigns or takes the address of the member
would then change only its own copy, and every later method call would read the original pointer, so such a struct
must keep the member.
"""

import re
from typing import Iterable

from ..text_scanners import mask_comments_and_literals

_THIS_ARROW = re.compile(r"\bthis\s*->\s*")
_THIS_DEREF = re.compile(r"\(\s*\*\s*this\s*\)\s*\.\s*")
_THIS_DOT = re.compile(r"\bthis\s*\.\s*")
# Assignment operators, longest first so that ``<<=`` is not read as ``<`` followed by ``<=``.
_ASSIGNMENTS = ("<<=", ">>=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=")


def pointer_member_is_written(body: str, member_names: Iterable[str]) -> bool:
    """Whether ``body`` reassigns, advances or takes the address of one of ``member_names``.

    ``member``, ``this->member``, ``(*this).member`` and ``this.member`` name this object's member; ``other.member``,
    ``p->member`` and ``Scope::member`` do not. A write through the member (``member[i] = x``, ``*member = x``) and a
    call to a function of the same name keep the pointer itself unchanged.
    """
    names = sorted({name for name in member_names if name})
    if not names or not body:
        return False
    text = mask_comments_and_literals(body)
    text = _THIS_DEREF.sub("", text)
    text = _THIS_ARROW.sub("", text)
    text = _THIS_DOT.sub("", text)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b"
    )
    for match in pattern.finditer(text):
        stop = match.start()
        while stop > 0 and text[stop - 1].isspace():
            stop -= 1
        before = text[max(0, stop - 3) : stop]
        if before.endswith((".", "->", "::")):
            continue
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        after = text[start : start + 3]
        if after.startswith("("):
            continue
        if before.endswith(("++", "--")):
            return True
        if (
            before.endswith("&")
            and not before.endswith("&&")
            and not after.startswith("[")
        ):
            return True
        if after.startswith(("++", "--")):
            return True
        # ``*member = x`` and ``*member += x`` write through the pointer; ``*member++`` still advances it.
        if before.endswith("*"):
            continue
        if after.startswith(_ASSIGNMENTS):
            return True
        if after.startswith("=") and not after.startswith("=="):
            return True
    return False

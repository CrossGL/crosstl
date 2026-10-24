"""Top-level delimiter scans over Metal source fragments that jump between the characters they act on.

The Metal preprocessor splits declarator lists, peels default initializers, finds parameter lists and strips
comments from fragments of C++ text many thousands of times per translation unit. Stepping through those
fragments one character at a time, with a ``str.startswith`` probe for comments at every position, was a large
share of the time spent on big kernels. These scans give the same results, but search for the next character
that can change their state, and return at once when the fragment cannot contain the delimiter they look for.

``read_string`` is the caller's quoted-literal reader: it returns the literal at a position and its length, so
these scans treat strings exactly as the scans they replaced did.
"""

import re
from typing import Callable, List, Optional, Tuple

ReadString = Callable[[str, int], Tuple[str, int]]

_COMMA_STOPS = re.compile(r"[\"'/<>()\[\]{},]")
_DEFAULT_VALUE_STOPS = re.compile(r"[\"'/<>()\[\]{}=]")
_PARAMETER_STOPS = re.compile(r"[\"'/<>()\[\]]")
_COMMENT_OR_QUOTE = re.compile(r"[\"']|//|/\*")
_COMPOUND_ASSIGNMENT_PREFIXES = "=<>!+-*/%&|^~"


def split_top_level_commas(text: str, read_string: ReadString) -> List[str]:
    """Split ``text`` at commas outside brackets, braces, parentheses, angles, strings and comments.

    Each part is stripped and an empty last part is dropped. ``<<`` never opens an angle.
    """
    if "," not in text:
        stripped = text.strip()
        return [stripped] if stripped else []
    parts: List[str] = []
    part_start = 0
    paren = bracket = brace = angle = 0
    i = 0
    length = len(text)
    search = _COMMA_STOPS.search
    while i < length:
        match = search(text, i)
        if match is None:
            break
        i = match.start()
        ch = text[i]
        if ch == '"' or ch == "'":
            i += read_string(text, i)[1]
            continue
        if ch == "/":
            if text.startswith("//", i):
                end = text.find("\n", i)
                if end == -1:
                    break
                i = end
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
        elif ch == "<" and text.startswith("<<", i):
            i += 2
            continue
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "<":
            if not (paren or bracket or brace):
                angle += 1
        elif ch == ">":
            if not (paren or bracket or brace):
                angle = max(0, angle - 1)
        elif not (paren or bracket or brace or angle):
            parts.append(text[part_start:i].strip())
            part_start = i + 1
        i += 1
    tail = text[part_start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _first_top_level_equals(
    text: str, read_string: ReadString, plain_assignment: bool
) -> Optional[int]:
    # Offset of the first `=` outside brackets, braces, parentheses, angles, strings and comments. A `//`
    # comment or an unterminated `/*` ends the scan. With ``plain_assignment`` the `=` must not be part of
    # `==` or of a comparison or compound assignment operator.
    paren = bracket = brace = angle = 0
    i = 0
    length = len(text)
    search = _DEFAULT_VALUE_STOPS.search
    while i < length:
        match = search(text, i)
        if match is None:
            return None
        i = match.start()
        ch = text[i]
        if ch == '"' or ch == "'":
            i += read_string(text, i)[1]
            continue
        if ch == "/":
            if text.startswith("//", i):
                return None
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif not (paren or bracket or brace or angle):
            if not plain_assignment:
                return i
            following = text[i + 1] if i + 1 < length else ""
            preceding = text[i - 1] if i > 0 else ""
            if following != "=" and preceding not in _COMPOUND_ASSIGNMENT_PREFIXES:
                return i
        i += 1
    return None


def strip_top_level_default_value(text: str, read_string: ReadString) -> str:
    """Return ``text`` up to its first top-level ``=``, stripped; all of ``text`` when it has none."""
    if "=" in text:
        equals = _first_top_level_equals(text, read_string, plain_assignment=False)
        if equals is not None:
            return text[:equals].strip()
    return text.strip()


def split_top_level_assignment(
    text: str, read_string: ReadString
) -> Tuple[str, Optional[str]]:
    """Split ``text`` at its first top-level plain ``=`` into stripped (lhs, rhs); rhs is None without one."""
    if "=" in text:
        equals = _first_top_level_equals(text, read_string, plain_assignment=True)
        if equals is not None:
            return text[:equals].strip(), text[equals + 1 :].strip()
    return text.strip(), None


def function_parameter_start(header: str, read_string: ReadString) -> Optional[int]:
    """Offset of the first ``(`` outside parentheses, brackets, angles, strings and comments in ``header``.

    An unterminated comment, or a ``//`` comment without a following line, ends the scan with None.
    """
    if "(" not in header:
        return None
    paren = bracket = angle = 0
    i = 0
    length = len(header)
    search = _PARAMETER_STOPS.search
    while i < length:
        match = search(header, i)
        if match is None:
            return None
        i = match.start()
        ch = header[i]
        if ch == '"' or ch == "'":
            i += read_string(header, i)[1]
            continue
        if ch == "/":
            if header.startswith("//", i):
                end = header.find("\n", i)
                if end == -1:
                    return None
                i = end + 1
                continue
            if header.startswith("/*", i):
                end = header.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
        elif ch == "(":
            if not (paren or bracket or angle):
                return i
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        i += 1
    return None


def strip_comments_outside_strings(value: str, read_string: ReadString) -> str:
    """Remove ``//`` and ``/* */`` comments from ``value``, keeping string literals verbatim.

    A ``//`` comment takes its newline with it. An unterminated comment, or a ``//`` comment without a
    following line, drops the rest of the text.
    """
    if "//" not in value and "/*" not in value:
        return value
    pieces: List[str] = []
    position = 0
    search = _COMMENT_OR_QUOTE.search
    while True:
        match = search(value, position)
        if match is None:
            pieces.append(value[position:])
            break
        start = match.start()
        pieces.append(value[position:start])
        token = match.group()
        if token == "//":
            end = value.find("\n", start)
            if end == -1:
                break
            position = end + 1
        elif token == "/*":
            end = value.find("*/", start + 2)
            if end == -1:
                break
            position = end + 2
        else:
            literal, consumed = read_string(value, start)
            pieces.append(literal)
            position = start + consumed
    return "".join(pieces)


_MASK_STARTS = re.compile(r"//|/\*|[\"']")
_NOT_NEWLINE = re.compile(r"[^\n]")


def mask_comments_and_literals(source: str) -> str:
    """Return ``source`` with comments and quoted literals blanked to spaces, keeping every offset.

    Newlines inside block comments and literals are kept. Inside a literal a backslash is blanked but the
    character it escapes is kept, so an escaped quote stays visible; a literal without its closing quote runs
    to the end of the text.
    """
    search = _MASK_STARTS.search
    match = search(source)
    if match is None:
        return source
    pieces: List[str] = []
    position = 0
    length = len(source)
    while match is not None:
        start = match.start()
        pieces.append(source[position:start])
        token = match.group()
        if token == "//":
            end = source.find("\n", start)
            if end == -1:
                end = length
            pieces.append(" " * (end - start))
            position = end
        elif token == "/*":
            end = source.find("*/", start + 2)
            stop = length if end == -1 else end + 2
            pieces.append(_NOT_NEWLINE.sub(" ", source[start:stop]))
            position = stop
        else:
            pieces.append(" ")
            i = start + 1
            while i < length:
                current = source[i]
                if current == "\\":
                    pieces.append(" ")
                    if i + 1 < length:
                        pieces.append(source[i + 1])
                    i += 2
                    continue
                pieces.append("\n" if current == "\n" else " ")
                i += 1
                if current == token:
                    break
            position = min(i, length)
        match = search(source, position)
    pieces.append(source[position:])
    return "".join(pieces)

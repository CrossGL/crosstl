"""Which function-local constants a template argument refers to, found in one pass over its text.

Template materialization refuses a specialization whose arguments still name a function-local constant without a
known value. Asking, for each such constant, whether substituting it changes an argument costs one scan of every
argument per constant, which is quadratic in a translation unit with thousands of local constants. These helpers
scan each argument once with the same lexical rules as
:meth:`MetalPreprocessor._substitute_template_argument_static_constants`, so they report exactly the constants that
substitution would replace.
"""

from typing import FrozenSet, Iterable, List, Sequence


def _read_string(text: str, start: int) -> int:
    """Length of the quoted literal at ``start``; an unterminated literal runs to the end of the text."""
    quote = text[start]
    i = start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i - start + 1
        i += 1
    return len(text) - start


def _read_identifier(text: str, start: int) -> int:
    i = start
    while i < len(text) and (text[i].isalnum() or text[i] == "_"):
        i += 1
    return i - start


def static_constant_references(argument: str) -> FrozenSet[str]:
    """Identifiers in ``argument`` that static-constant substitution would replace if they were constants.

    An identifier qualifies unless it follows ``.`` or ``::`` (a member or a qualified name) or precedes ``(`` (a
    call). String and character literals and comments are skipped.
    """
    found = set()
    i = 0
    length = len(argument)
    while i < length:
        character = argument[i]
        if character in "\"'":
            i += _read_string(argument, i)
            continue
        if argument.startswith("//", i):
            break
        if argument.startswith("/*", i):
            end = argument.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue
        if character.isalpha() or character == "_":
            consumed = _read_identifier(argument, i)
            previous = i - 1
            while previous >= 0 and argument[previous].isspace():
                previous -= 1
            scoped = previous >= 0 and argument[previous] == "."
            if previous >= 0 and argument[previous] == ":":
                qualifier_colon = previous - 1
                while qualifier_colon >= 0 and argument[qualifier_colon].isspace():
                    qualifier_colon -= 1
                scoped = qualifier_colon >= 0 and argument[qualifier_colon] == ":"
            following = i + consumed
            while following < length and argument[following].isspace():
                following += 1
            if not scoped and (following >= length or argument[following] != "("):
                found.add(argument[i : i + consumed])
            i += consumed
            continue
        i += 1
    return frozenset(found)


def unresolved_constant_references(
    arguments: Sequence[str], unresolved_names: Iterable[str]
) -> List[str]:
    """The sorted names in ``unresolved_names`` that substitution would change some argument for.

    Substitution also strips the argument, so an argument with surrounding whitespace counts as changed for every
    name, as it does for the per-name substitution this replaces.
    """
    names = list(unresolved_names)
    if any(argument != argument.strip() for argument in arguments):
        return sorted(names)
    referenced = set()
    for argument in arguments:
        referenced |= static_constant_references(argument)
    return sorted(name for name in names if name in referenced)

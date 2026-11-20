"""Reading Mojo source text: the names a package's ``__init__.mojo`` imports, the module-level definitions of a
``.mojo`` file and the methods of one of its structs, each split into its compile-time parameters, arguments, result
and body. :mod:`.mojo` builds the Mojo feature list from them.
"""

from __future__ import annotations

import re

_EXPORTS = re.compile(r"^from\s+(\S+)\s+import\s+(\([^)]*\)|[^\n]+)", re.M)
_DEFINITION = re.compile(r"^def\s+(\w+)\s*", re.M)
_METHOD = re.compile(r"^((?:@[^\n]*\n)*)def\s+(\w+)", re.M)

Definition = tuple[str, str, str, str]


def exports(init: str, module: str, functions: bool = True) -> list[str]:
    """The function names a package's ``__init__.mojo`` imports from ``module``, in file order; every name it imports
    unless ``functions``."""
    names = []
    for source, imported in _EXPORTS.findall(init):
        if source != module:
            continue
        for name in re.split(r"[\s,()]+", imported):
            if name and (name[0].islower() or not functions) and name not in names:
                names.append(name)
    return names


def split_commas(text: str) -> list[str]:
    """``text`` split at its top-level commas."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _group(text: str, start: int, closing: str) -> tuple[str, int]:
    """The text inside the bracket group that opens at ``text[start]``, and the index just past it."""
    opening, depth = text[start], 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError(f"unbalanced {opening}{closing} at offset {start}")


def definitions(text: str) -> dict[str, Definition]:
    """``name -> (parameters, arguments, result, body)`` for the first module-level definition of each name.

    ``body`` starts with the header's ``where`` clause, so a constraint written there or asserted in the body is
    found the same way.
    """
    found: dict[str, Definition] = {}
    for match in _DEFINITION.finditer(text):
        name, position = match.group(1), match.end()
        if name in found:
            continue
        parameters = ""
        if text[position] == "[":
            parameters, position = _group(text, position, "]")
        arguments, position = _group(text, position, ")")
        header_end = text.index(":\n", position)
        tail = text[position:header_end]
        result = re.sub(r"\swhere\s.*", "", tail, flags=re.S).strip()
        following = re.search(r"^(?=[^\s#])", text[header_end + 2 :], re.M)
        body_end = header_end + 2 + (following.start() if following else len(text))
        found[name] = (
            parameters,
            arguments,
            result.removeprefix("->").strip(),
            tail + text[header_end:body_end],
        )
    return found


def compile_parameters(text: str) -> list[tuple[str, str, bool, str]]:
    """``(name, kind, required, default)`` per compile-time parameter.

    Parameters before ``//`` are inferred from the arguments, and a parameter with a default needs no value, so
    neither is required.
    """
    parts = split_commas(re.sub(r"#[^\n]*", "", text))
    inferred = "//" in parts
    result = []
    for part in parts:
        if part == "//":
            inferred = False
            continue
        if part == "*":
            continue
        name, _, rest = part.partition(":")
        kind, _, default = rest.partition("=")
        result.append(
            (
                name.strip(),
                kind.strip(),
                not inferred and not default.strip(),
                default.strip(),
            )
        )
    return result


def argument_default(part: str) -> tuple[str, str]:
    """An argument's declaration and its default value, split at the first ``=`` outside brackets, as in
    ``ptr: MutPointer[Scalar[dtype], _, address_space=_]``, which has none."""
    depth = 0
    for index, char in enumerate(part):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            return part[:index], part[index + 1 :]
    return part, ""


def methods(
    text: str, struct: str
) -> tuple[dict[str, Definition], dict[str, Definition]]:
    """The definitions of the public static methods of the struct ``struct`` by name, and those of the public methods
    only its values have, in file order; an instance method's arguments start with ``self``.
    """
    found = re.search(rf"^struct\s+{struct}\b", text, re.M)
    if not found:
        return {}, {}
    position = found.end()
    if text[position] == "[":
        _, position = _group(text, position, "]")
    header_end = text.index(":\n", position) + 2
    following = re.search(r"^(?=[^\s#])", text[header_end:], re.M)
    end = header_end + (following.start() if following else len(text))
    body = re.sub(r"^    ", "", text[header_end:end], flags=re.M)
    static: dict[str, Definition] = {}
    instance: dict[str, Definition] = {}
    for match in _METHOD.finditer(body):
        decorators, name = match.groups()
        if name.startswith("_"):
            continue
        kept = static if "@staticmethod" in decorators else instance
        kept.setdefault(name, definitions(body[match.start(2) - 4 :])[name])
    return static, {n: d for n, d in instance.items() if n not in static}


def public(text: str) -> list[str]:
    """The names of a module's module-level definitions that do not start with an underscore, in file order."""
    names = []
    for name in _DEFINITION.findall(text):
        if not name.startswith("_") and name not in names:
            names.append(name)
    return names

"""Structs of the same name declared in several namespaces of a Metal unit, as ``ops::P`` and ``alt::P``.

CrossGL has one global scope, so the importer emits every namespace member at program scope under its unqualified
name, and the struct passes know a struct by that name. Two structs ``ops::P`` and ``alt::P`` then became one
``P``: their methods, static members and overloads taking them could not be told apart, and the import failed.

:func:`distinct_struct_homonyms` gives each such struct, except one declared in the global namespace, a name of its
own that occurs nowhere else in the unit, ``ops_P`` for ``ops::P``, and renames every reference to it the way C++
name lookup binds the reference ([basic.lookup.qual], [basic.lookup.unqual], [namespace.udir]). The rename keeps
every qualifier and namespace, so the text stays a valid unit that means what the source means, and the struct
passes then see distinct names.

A name the rename cannot follow safely keeps its spelling, so the later passes still fail closed on it: one that
is also declared at block or class scope, in an unnamed namespace, as an alias, an enumeration or a function, one
used as a member name after ``.`` or ``->``, and one with an out-of-class member definition such as ``P::P``. A
reference whose lookup finds structs of that name in more than one namespace is ambiguous in C++ too and keeps its
spelling, and the import refuses it with :class:`AmbiguousStructNameError`; one that owns a qualified member, as
``Limits::max``, keeps the name's spelling everywhere, so the static-member passes refuse it with the owner and member.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .namespaced_functions import nominated_namespaces_in_scope
from .text_scanners import mask_comments_and_literals

__all__ = ["AmbiguousStructNameError", "distinct_struct_homonyms"]


class AmbiguousStructNameError(TranslationDiagnostic):
    """An unqualified struct name whose lookup finds structs of that name in more than one namespace."""

    code = f"{CODE_PREFIX}metal-ambiguous-struct-name"


class _AmbiguousOwner(Exception):
    """An ambiguous name used as the owner of a qualified member, as in ``Limits::max``."""


_IDENT = r"[A-Za-z_]\w*"
_NAMESPACE_OPEN = re.compile(
    rf"\b(?:inline\s+)?namespace\b\s*(?P<name>(?:{_IDENT}\s*::\s*)*{_IDENT})?\s*(?:\[\[[^\]]*\]\]\s*)?\{{"
)
_STRUCT_DECLARATION = re.compile(
    rf"\b(?:struct|class|union)\s+(?:alignas\s*\([^)]*\)\s*)?(?:\[\[[^\]]*\]\]\s*)?(?P<name>{_IDENT})"
    r"\s*(?:final\s*)?(?=[:{;])(?!::)"
)
_USING_DIRECTIVE = re.compile(
    rf"\busing\s+namespace\s+(?P<target>(?:::\s*)?(?:{_IDENT}\s*::\s*)*{_IDENT})\s*;"
)
# Words that may precede a constructor or a call to a struct's constructor, as opposed to a function's return type.
_NOT_A_RETURN_TYPE = frozenset(
    {"return", "case", "else", "do", "explicit", "constexpr", "inline", "throw"}
)


@dataclass
class _Frame:
    start: int
    end: int
    namespace: None | (
        str
    )  # the full namespace name of a namespace body, None for any other brace block
    parent: int  # the index of the enclosing frame, -1 at file scope


def _frames(masked: str) -> list[_Frame]:
    """Every brace block of ``masked`` in source order, each with its enclosing block."""
    opens = {match.end() - 1: match for match in _NAMESPACE_OPEN.finditer(masked)}
    frames: list[_Frame] = []
    stack: list[int] = []
    for brace in re.finditer(r"[{}]", masked):
        position = brace.start()
        if brace.group() == "}":
            if stack:
                frames[stack.pop()].end = position + 1
            continue
        parent = stack[-1] if stack else -1
        namespace = None
        match = opens.get(position)
        if match is not None:
            name = re.sub(r"\s+", "", match.group("name") or "") or "<unnamed>"
            outer = _namespace_of(frames, parent)
            namespace = f"{outer}::{name}" if outer else name
        stack.append(len(frames))
        frames.append(_Frame(position, len(masked), namespace, parent))
    return frames


def _namespace_of(frames: list[_Frame], index: int) -> str:
    while index >= 0:
        if frames[index].namespace is not None:
            return frames[index].namespace
        index = frames[index].parent
    return ""


class _Scopes:
    def __init__(self, masked: str):
        self.frames = _frames(masked)
        self.starts = [frame.start for frame in self.frames]
        self.namespaces = {f.namespace for f in self.frames if f.namespace}

    def _innermost_index(self, position: int) -> int:
        index = bisect_right(self.starts, position) - 1
        while index >= 0 and self.frames[index].end <= position:
            index = self.frames[index].parent
        return index

    def innermost(self, position: int) -> _Frame | None:
        index = self._innermost_index(position)
        return self.frames[index] if index >= 0 else None

    def namespace_at(self, position: int) -> str:
        return _namespace_of(self.frames, self._innermost_index(position))

    def resolve(self, spelled: str, origin: str) -> str | None:
        """The namespace ``spelled`` names when looked up from namespace ``origin``."""
        spelled = re.sub(r"\s+", "", spelled)
        if spelled.startswith("::"):
            return spelled[2:] if spelled[2:] in self.namespaces else None
        parts = origin.split("::") if origin else []
        for depth in range(len(parts), -1, -1):
            prefix = "::".join(parts[:depth])
            candidate = f"{prefix}::{spelled}" if prefix else spelled
            if candidate in self.namespaces:
                return candidate
        return None


def _enclosing(namespace: str) -> list[str]:
    parts = namespace.split("::") if namespace else []
    return ["::".join(parts[:depth]) for depth in range(len(parts), -1, -1)]


def _unsafe(name: str, masked: str, scopes: _Scopes, block_scoped: set[str]) -> bool:
    word = re.escape(name)
    if name in block_scoped:
        return True
    if re.search(rf"(?:\.|->)\s*{word}\b", masked):
        return True
    if re.search(rf"\b{word}\s*::\s*~?\s*{word}\b|~\s*{word}\b", masked):
        return True
    if re.search(
        rf"\busing\s+{word}\s*=|\benum\s+(?:class\s+|struct\s+)?{word}\b", masked
    ):
        return True
    if re.search(rf"\btypedef\b[^;{{}}]*\b{word}\s*;", masked):
        return True
    for match in re.finditer(rf"\b(?P<word>{_IDENT})[\s*&]+{word}\s*\(", masked):
        if match.group("word") not in _NOT_A_RETURN_TYPE:
            return True
    return False


def _qualifier(masked: str, start: int) -> tuple[str | None, bool]:
    """The ``a::b`` qualifier spelled before ``start`` (``None`` when it is not a namespace path) and whether the
    reference is qualified at all."""
    parts: list[str] = []
    cursor = start
    anchored = False
    while True:
        before = masked[:cursor].rstrip()
        if not before.endswith("::"):
            break
        before = before[:-2].rstrip()
        found = re.search(rf"{_IDENT}$", before)
        if found is None:
            if before.endswith(">"):
                return None, True
            anchored = True
            break
        parts.insert(0, found.group())
        cursor = found.start()
    if not parts and not anchored:
        return "", False
    return ("::" if anchored else "") + "::".join(parts), True


def distinct_struct_homonyms(code: str) -> str:
    """``code`` with each struct whose name another namespace also declares renamed to a name of its own."""
    if "namespace" not in code:
        return code
    masked = mask_comments_and_literals(code)
    scopes = _Scopes(masked)
    declared: dict[str, set[str]] = {}
    block_scoped: set[str] = set()
    for match in _STRUCT_DECLARATION.finditer(masked):
        frame = scopes.innermost(match.start())
        name = match.group("name")
        if frame is not None and frame.namespace is None:
            block_scoped.add(name)
            continue
        declared.setdefault(name, set()).add(scopes.namespace_at(match.start()))
    homonyms = {
        name: namespaces
        for name, namespaces in declared.items()
        if len(namespaces) > 1
        and not any("<unnamed>" in namespace for namespace in namespaces)
        and not _unsafe(name, masked, scopes, block_scoped)
    }
    if not homonyms:
        return code

    taken = set(re.findall(_IDENT, masked))
    renamed: dict[tuple[str, str], str] = {}
    for name in sorted(homonyms):
        for namespace in sorted(homonyms[name]):
            if not namespace:
                continue
            candidate = f"{namespace.replace('::', '_')}_{name}"
            while candidate in taken:
                candidate += "_"
            taken.add(candidate)
            renamed[(namespace, name)] = candidate

    directives = []
    for match in _USING_DIRECTIVE.finditer(masked):
        frame = scopes.innermost(match.start())
        origin = scopes.namespace_at(match.start())
        target = scopes.resolve(match.group("target"), origin)
        if target is not None:
            directives.append((match.start(), origin, frame, target))

    def nominated(position: int):
        def nominated_by(namespace: str) -> list[str]:
            return [
                target
                for at, origin, frame, target in directives
                if origin == namespace
                and at < position
                and (
                    frame is None
                    or frame.namespace is not None
                    or frame.start <= position < frame.end
                )
            ]

        return nominated_by

    def lookup(name: str, position: int, qualifier: str, qualified: bool) -> str | None:
        origin = scopes.namespace_at(position)
        if qualified:
            namespace = scopes.resolve(qualifier, origin) if qualifier != "::" else ""
            if namespace is None or namespace not in homonyms[name]:
                return None
            return namespace
        nominated_by = nominated(position)
        for scope in _enclosing(origin):
            found = {scope} if scope in homonyms[name] else set()
            found.update(
                namespace
                for namespace in nominated_namespaces_in_scope(
                    origin, scope, nominated_by
                )
                if namespace in homonyms[name]
            )
            if len(found) > 1:
                if re.match(r"\s*::", masked[position + len(name) :]):
                    # `Limits::max` through an ambiguous owner: the static-member passes name the owner and member.
                    raise _AmbiguousOwner(name)
                line = code.count("\n", 0, position) + 1
                owners = " and ".join(
                    f"{namespace}::{name}" for namespace in sorted(found)
                )
                raise AmbiguousStructNameError(
                    f"line {line}: the unqualified name '{name}' is ambiguous: lookup finds {owners} "
                    "([namespace.udir])"
                )
            if found:
                return found.pop()
        return None

    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(homonyms))) + r")\b")
    edits: list[tuple[int, int, str]] = []
    kept: set[str] = set()
    for match in pattern.finditer(masked):
        qualifier, qualified = _qualifier(masked, match.start())
        if qualifier is None or match.group() in kept:
            continue
        try:
            namespace = lookup(match.group(), match.start(), qualifier, qualified)
        except _AmbiguousOwner:
            kept.add(match.group())
            continue
        replacement = (
            renamed.get((namespace, match.group())) if namespace is not None else None
        )
        if replacement is not None:
            edits.append((match.start(), match.end(), replacement))
    # A name used as an ambiguous owner keeps its spelling everywhere, so the later passes see the source's structs.
    edits = [edit for edit in edits if code[edit[0] : edit[1]] not in kept]
    for start, end, replacement in reversed(edits):
        code = code[:start] + replacement + code[end:]
    return code

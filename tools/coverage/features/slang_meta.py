"""Read the declarations of Slang's core module sources, ``hlsl.meta.slang`` and ``core.meta.slang``.

The files are templates: the lines between ``${`` and ``$}`` are C++ that generates the module, and ``$(expr)`` splices
a value into it. :func:`declarations` takes every global function in the sections the files mark ``//@public:``, the
sections Slang's own reference documents. A name a meta loop generates, such as
``Interlocked$(atomicOp.slangCallSuffix)``, is expanded over the loop's table. Member functions of resource and object
types are left out, and so are functions marked ``internal``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MODIFIERS = frozenset(
    "public static inline internal export __extern_cpp __unsafeForceInlineEarly __noinline nodiff".split()
)
_LEADING_CALL = re.compile(
    r"(__generic|__glsl_extension|__glsl_version|__spirv_version|__cuda_sm_version|__wgsl_extension|"
    r"__target_intrinsic|__intrinsic_op|__specialized_for_target)\s*(?=[(<])"
)
HEAD = re.compile(
    r"(?P<result>[A-Za-z_][\w:]*(?:\s*<.*>)?(?:\s*\[\s*\d*\s*\])?)\s+(?P<name>[A-Za-z_$][\w$().]*)\s*"
    r"(?P<generic><[^()]*>)?\s*\((?P<params>.*)\)\s*(?:->\s*(?P<arrow>.+))?$",
    re.S,
)
_LOOP = re.compile(
    r"for\s*\(\s*(?:const\s+)?(?:auto|[\w:]+)\s*&?\s*(\w+)\s*:\s*(\w+)\s*\)"
)
_STRUCT = re.compile(r"struct\s+(\w+)\s*\{([^}]*)\}")
_TABLE = re.compile(
    r"(?:(?:static|const)\s+)*(?:struct\s*\{(?P<fields>[^}]*)\}|(?P<type>\w+))\s*(?:const\s*)?\*?\s*"
    r"(?P<name>\w+)\s*\[[^\]]*\]\s*=\s*\{(?P<body>.*?)\}\s*;",
    re.S,
)
_SPLICE = re.compile(r"\$\(\s*(\w+)(?:\.(\w+))?\s*\)")


@dataclass(frozen=True)
class Declaration:
    """One overload: its name, its head (``T sin(T x)``), its generic parameters and its ``require`` attribute."""

    name: str
    head: str
    generics: str
    requires: str = ""


def _strip_block_comments(text: str) -> str:
    return re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S
    )


def _code(line: str) -> str:
    """A code line with strings and character literals emptied and its ``//`` comment removed."""
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])'", "''", line)
    return line.split("//")[0]


def _fields(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"(\w+)\s*(?:\[[^\]]*\])?\s*;", text)]


def _items(text: str) -> list[str]:
    """Split a C++ initializer list on its top-level commas."""
    items, depth, current = [], 0, ""
    for char in text:
        depth += {"{": 1, "(": 1, "}": -1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += char
    return [item for item in [*items, current.strip()] if item]


class _Meta:
    """The tables and loop variables of the meta code read so far."""

    def __init__(self):
        self.structs: dict[str, list[str]] = {}
        self.tables: dict[str, list[dict[str, str]]] = {}
        self.loops: dict[str, str] = {}

    def read(self, code: str) -> None:
        code = re.sub(r"//[^\n]*", "", code)
        for match in _STRUCT.finditer(code):
            self.structs[match.group(1)] = _fields(match.group(2))
        for match in _TABLE.finditer(code):
            names = (
                _fields(match.group("fields"))
                if match.group("fields") is not None
                else self.structs.get(match.group("type"), [])
            )
            self.tables[match.group("name")] = [
                self._entry(item, names) for item in _items(match.group("body"))
            ]
        for variable, table in _LOOP.findall(code):
            self.loops[variable] = table

    @staticmethod
    def _entry(item: str, names: list[str]) -> dict[str, str]:
        values = _items(item[1:-1]) if item.startswith("{") else [item]
        values = [v.strip().strip('"') for v in values]
        entry = dict(zip(names, values))
        entry[""] = values[0] if values else ""
        return entry

    def expand(self, text: str) -> list[str] | None:
        """``text`` once per entry of the table its splices loop over; ``None`` when a splice is not a loop value."""
        variables = {m.group(1) for m in _SPLICE.finditer(text)}
        if not variables:
            return [text]
        if len(variables) > 1 or not variables <= self.loops.keys():
            return None
        table = self.tables.get(self.loops[next(iter(variables))])
        if not table:
            return None
        expanded = []
        for entry in table:

            def value(match: re.Match, entry=entry) -> str:
                return entry.get(match.group(2) or "", "")

            expanded.append(_SPLICE.sub(value, text))
        return expanded


def _strip_leading(head: str) -> tuple[str, list[str], list[str]]:
    """The head without leading attributes and modifiers, the attributes, and the ``__generic`` parameter lists."""
    attributes, generics = [], []
    head = head.strip()
    while True:
        if head.startswith("["):
            end = _closing(head, 0, "[]")
            attributes.append(head[1:end])
            head = head[end + 1 :].lstrip()
            continue
        call = _LEADING_CALL.match(head)
        if call:
            pair = "<>" if head[call.end()] == "<" else "()"
            end = _closing(head, call.end(), pair)
            if call.group(1) == "__generic":
                generics.append(head[call.end() + 1 : end])
            head = head[end + 1 :].lstrip()
            continue
        word = re.match(r"(\w+)\s+", head)
        if word and word.group(1) in _MODIFIERS:
            head = head[word.end() :]
            continue
        return head, attributes, generics


def declarations(text: str) -> list[Declaration]:
    """Every global function the public sections of a ``.meta.slang`` file declare, meta-loop names expanded."""
    meta, found = _Meta(), []
    depth, public, in_meta = 0, False, False
    meta_lines: list[str] = []
    head = ""
    for line in _strip_block_comments(text).splitlines():
        stripped = line.strip()
        if in_meta or stripped.startswith("${"):
            body = line if in_meta else stripped[2:]
            in_meta = "$}" not in body
            meta_lines.append(body.split("$}")[0])
            if not in_meta:
                meta.read("\n".join(meta_lines))
                meta_lines = []
            continue
        marker = re.match(r"\s*//\s*@\s*(public|hidden)\s*:", line)
        if marker and depth == 0:
            public = marker.group(1) == "public"
            continue
        code, index = _code(line), 0
        while index < len(code):
            char = code[index]
            if code.startswith("$(", index):
                end = _closing(code, index + 1)
                if depth == 0:
                    head += code[index : end + 1]
                index = end + 1
                continue
            index += 1
            if char in "{;" and depth == 0:
                # A function is defined (``{``) or, for an intrinsic, only declared (``;``).
                if public:
                    found += _declarations(meta, head)
                depth += char == "{"
                head = ""
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                head = ""
            elif depth == 0:
                head += char
        head += " "
    return found


def _closing(text: str, start: int, pair: str = "()") -> int:
    """The index of the bracket that closes the one at ``start``, or the last index when it is not closed."""
    depth = 0
    for index in range(start, len(text)):
        depth += {pair[0]: 1, pair[1]: -1}.get(text[index], 0)
        if depth == 0:
            return index
    return len(text) - 1


def _declarations(meta: _Meta, raw: str) -> list[Declaration]:
    raw = " ".join(raw.split())
    head, attributes, generics = _strip_leading(raw)
    # An ``internal`` function is not part of the public module, whichever section declares it. A function
    # extension, such as ``fwd_diff(CoopVec<T, N>::__subscript::get)``, gives an existing function a derivative and
    # declares no function of its own.
    if re.search(r"\binternal\b", raw[: len(raw) - len(head)]) or head.startswith(
        "__func_extension"
    ):
        return []
    requires = " ".join(a for a in attributes if a.startswith("require"))
    # Splices may appear in the name, the types and the constraints, so the three are expanded together.
    variants = meta.expand("\0".join([head, requires, ", ".join(generics)]))
    result = []
    for variant in variants or []:
        head, requires, generic = variant.split("\0")
        match = HEAD.fullmatch(head)
        if not match or match.group("result") in ("return", "typedef", "typealias"):
            continue
        name = match.group("name")
        if name.startswith("_") or "$" in name or name.startswith("operator"):
            continue
        inline = (match.group("generic") or "").strip("<> ")
        generic = ", ".join(g for g in (generic, inline) if g)
        result.append(Declaration(name, head, generic, requires))
    return result

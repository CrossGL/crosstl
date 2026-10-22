"""WGSL built-in functions from Tint's intrinsic definitions (``src/tint/lang/wgsl/wgsl.def`` in google/dawn).

Only the "Builtin Functions" section is read; value constructors and operators follow it. Sections and functions
that exist only as Chromium extensions are left out. Template parameters are made concrete the way the GLSL list
does it: a vector of any width becomes a three-component vector, a numeric type its 32-bit float member, and an
explicit template argument (``bitcast<T>``) its unsigned member, so a cast changes the type it casts.
"""

from __future__ import annotations

import re

from .. import spellings
from ..model import VOID, Feature, Param, Signature, Type
from ..objects import UNIFORMS
from .common import pick, prototype, split

LEFT_OUT = {
    "subgroup matrix": "Chromium extension (chromium_experimental_subgroup_matrix)",
    "buffer view": "Chromium extension",
}
LEFT_OUT_NAMES = {
    "print": "Chromium extension (chromium_experimental_print)",
    "atomicStoreMax": "Chromium extension (atomic_vec2u_min_max)",
    "atomicStoreMin": "Chromium extension (atomic_vec2u_min_max)",
    "inputAttachmentLoad": "Chromium extension (chromium_internal_input_attachments)",
}
# Operations CrossGL has no spelling for, with the reason.
UNSPELLED = {
    "textureSampleBaseClampToEdge": (
        "no CrossGL sample clamps its coordinates half a texel inside the edge"
    ),
}
_SCALARS = {"f32": "float", "i32": "int", "u32": "uint", "bool": "bool"}
_IMPLICIT_ORDER = ("f32", "i32", "u32", "bool")
_EXPLICIT_ORDER = ("u32", "i32", "f32", "bool")
_SIZE = 3
_ATTRIBUTE = re.compile(r"@(\w+)(?:\(([^()]*)\))?")
_DECLARATION = re.compile(
    r"(?P<attributes>(?:@\w+(?:\([^()]*\))?\s*)*)"
    r"(?:implicit\((?P<implicit>[^()]*)\)\s*)?"
    r"fn\s+(?P<name>\w+)\s*(?:<(?P<template>.*?)>\s*)?\((?P<params>.*)\)\s*(?:->\s*(?P<result>.+))?",
    re.S,
)
_MATCHER = re.compile(r"match\s+(\w+)\s*:\s*(.+)")
_GENERIC = re.compile(r"(\w+)\s*<(.*)>")


class _Types:
    """Resolve type spellings of one declaration, given the file's matchers and the declaration's bindings."""

    def __init__(self, matchers: dict[str, list[str]]):
        self.matchers = matchers
        self.bindings: dict[str, Type | int] = {}

    def bind(self, declarations: str, explicit: bool) -> None:
        order = _EXPLICIT_ORDER if explicit else _IMPLICIT_ORDER
        for item in split(declarations):
            name, _, constraint = (part.strip() for part in item.partition(":"))
            if constraint == "num":
                self.bindings[name] = _SIZE
            elif not constraint:
                self.bindings[name] = Type("float")
            elif name == "I" and constraint == "iu32":
                # A lane index; naga accepts only u32 there.
                self.bindings[name] = Type("uint")
            elif constraint == "__constructible":
                # Any concrete constructible type, such as the one workgroupUniformLoad reads.
                self.bindings[name] = Type("float")
            elif constraint in self.matchers:
                self.bindings[name] = self._choose(self.matchers[constraint], order)
            else:
                self.bindings[name] = self.type(constraint, order)

    def _choose(self, members: list[str], order: tuple[str, ...]) -> Type:
        for scalar in order:
            if scalar in members:
                return Type(_SCALARS[scalar])
        return Type.other(" | ".join(members))

    def _size(self, text: str) -> int:
        bound = self.bindings.get(text.strip())
        if isinstance(bound, int):
            return bound
        return int(text) if text.strip().isdigit() else _SIZE

    def type(self, spelling: str, order: tuple[str, ...] = _IMPLICIT_ORDER) -> Type:
        spelling = spelling.strip()
        bound = self.bindings.get(spelling)
        if isinstance(bound, Type):
            return bound
        if spelling in _SCALARS:
            return Type(_SCALARS[spelling])
        if spelling in self.matchers:
            return self._choose(self.matchers[spelling], order)
        generic = _GENERIC.fullmatch(spelling)
        if not generic:
            return Type.other(spelling)
        base, arguments = generic.group(1), split(generic.group(2))
        vector = re.fullmatch(r"vec([2-4])?", base)
        matrix = re.fullmatch(r"mat(?:([2-4])x([2-4]))?", base)
        if vector and len(arguments) == (1 if vector.group(1) else 2):
            size = int(vector.group(1)) if vector.group(1) else self._size(arguments[0])
            element = self.type(arguments[-1], order)
            return element if element.opaque else Type(element.scalar, (size,))
        if matrix and len(arguments) == (1 if matrix.group(1) else 3):
            if matrix.group(1):
                shape = (int(matrix.group(1)), int(matrix.group(2)))
            else:
                shape = (self._size(arguments[0]), self._size(arguments[1]))
            element = self.type(arguments[-1], order)
            if element.opaque or element.scalar != "float":
                return Type.other(spelling)
            return Type(element.scalar, shape)
        return Type.other(spelling)

    def param(self, text: str) -> Param:
        constant = any(name == "const" for name, _ in _ATTRIBUTE.findall(text))
        text = _ATTRIBUTE.sub("", text).strip()
        text = re.sub(r"^\w+\s*:\s*", "", text)
        pointer = re.fullmatch(r"ptr\s*<(.*)>", text)
        if pointer:
            arguments = split(pointer.group(1))
            atomic = (
                re.fullmatch(r"atomic\s*<(.*)>", arguments[1])
                if len(arguments) > 1
                else None
            )
            # The probe's atomic operand is an element of a storage buffer.
            if atomic and arguments[0] != "workgroup":
                return Param(self.type(atomic.group(1)), "inout")
            # A pointer to a workgroup variable of a value type, such as workgroupUniformLoad reads.
            if not atomic and arguments[0] == "workgroup" and len(arguments) > 1:
                kind = self.type(arguments[1])
                if not kind.opaque:
                    return Param(kind, "workgroup")
            # A pointer to a runtime-sized array, such as arrayLength reads: the probe's output buffer.
            if len(arguments) > 1 and re.fullmatch(
                r"runtime_array\s*<.*>", arguments[1]
            ):
                return Param(Type("uint"), "buffer")
            return Param(Type.other(text))
        return Param(self.type(text), "const" if constant else "in")


def matchers(text: str) -> dict[str, list[str]]:
    found = {}
    for line in text.splitlines():
        match = _MATCHER.fullmatch(line.strip())
        if match:
            found[match.group(1)] = [m.strip() for m in match.group(2).split("|")]
    return found


def declarations(text: str):
    """Yield ``(group, declaration text)`` for every built-in function declaration."""
    lines = text.splitlines()
    start = next(n for n, line in enumerate(lines) if "// Builtin Functions" in line)
    group, pending, heading = "builtin functions", "", ""
    for line in lines[start + 1 :]:
        if "// Value constructors" in line:
            break
        stripped = line.strip()
        # A section heading is one comment line between two banner lines.
        if re.fullmatch(r"/{6,}", stripped):
            if heading:
                group, heading = heading, ""
            else:
                heading = "\0"
            continue
        named = re.fullmatch(r"//\s*([A-Za-z][\w ]*)", stripped)
        heading = named.group(1).strip().lower() if heading == "\0" and named else ""
        code = stripped.split("//")[0].strip()
        if not code:
            continue
        pending = f"{pending} {code}".strip()
        # A declaration ends on the line that closes its parameter list; the result type shares that line.
        if re.search(r"\bfn\s", pending) and pending.count("(") == pending.count(")"):
            yield group, pending
            pending = ""


def _stage(attributes: str) -> str:
    for name, arguments in _ATTRIBUTE.findall(attributes):
        if name == "stage":
            stages = re.findall(r'"(\w+)"', arguments)
            return "compute" if "compute" in stages else stages[0]
    return "compute"


def _overload(
    declaration: str, found: dict[str, list[str]]
) -> tuple[str, str, str, Signature] | None:
    match = _DECLARATION.fullmatch(declaration)
    if not match:
        return None
    types = _Types(found)
    if match.group("implicit"):
        types.bind(match.group("implicit"), explicit=False)
    template = ""
    if match.group("template"):
        types.bind(match.group("template"), explicit=True)
        names = [
            item.partition(":")[0].strip() for item in split(match.group("template"))
        ]
        spelled = [types.bindings.get(n) for n in names]
        template = ",".join(
            _wgsl(kind) if isinstance(kind, Type) else "?" for kind in spelled
        )
    params = tuple(types.param(p) for p in split(match.group("params")))
    result = (
        types.type(match.group("result"), _EXPLICIT_ORDER)
        if match.group("result")
        else VOID
    )
    return (
        match.group("name"),
        _stage(match.group("attributes")),
        template,
        Signature(result, params),
    )


def _wgsl(kind: Type) -> str:
    """The WGSL spelling of a template argument (used only for explicit templates such as ``bitcast<u32>``)."""
    if kind.opaque:
        return kind.opaque
    names = {v: k for k, v in _SCALARS.items()}
    if not kind.shape:
        return names[kind.scalar]
    return f"vec{kind.shape[0]}<{names[kind.scalar]}>"


def _scalar_only(signature: Signature) -> bool:
    return not any(len(p.type.shape) == 1 for p in signature.params)


def parse(
    text: str, glsl_names: set[str] | frozenset[str] = frozenset()
) -> list[Feature]:
    found = matchers(text)
    functions: dict[str, list[tuple[str, str, str, Signature]]] = {}
    for group, declaration in declarations(text):
        if group in LEFT_OUT:
            continue
        overload = _overload(declaration, found)
        if overload and overload[0] not in LEFT_OUT_NAMES:
            functions.setdefault(overload[0], []).append((group, *overload[1:]))
    features = []
    for name, overloads in sorted(functions.items()):
        # Vector overloads first, as the GLSL list makes its generic types vectors.
        chosen = pick(
            sorted((o[3] for o in overloads), key=_scalar_only), UNIFORMS["wgsl"]
        )
        group, stage, template, _ = next(o for o in overloads if o[3] is chosen)
        spelling, counterpart = spellings.spelling("wgsl", name, glsl_names), None
        if name in spellings.COUNTERPARTS["wgsl"]:
            spelling, counterpart = prototype(spellings.COUNTERPARTS["wgsl"][name])
        if name in UNSPELLED:
            spelling = ""
        features.append(
            Feature(
                "wgsl",
                "function",
                name,
                chosen,
                group=group,
                crossgl=spelling,
                stage=stage,
                spelling=f"{name}<{template}>" if template else "",
                note=UNSPELLED.get(name, ""),
                counterpart=counterpart,
            )
        )
    return features

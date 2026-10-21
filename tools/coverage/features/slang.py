"""Slang built-in functions from the core module sources slangc is built from.

``hlsl.meta.slang`` and ``core.meta.slang`` define Slang's standard library. They are templates: the lines between
``${`` and ``$}`` are C++ that generates the module, and ``$(expr)`` splices a value into it. The list takes every
global function in the sections the files mark ``//@public:``, the sections Slang's own reference documents. A name a
meta loop generates, such as ``Interlocked$(atomicOp.slangCallSuffix)``, is expanded over the loop's table. Member
functions of resource and object types are left out.

Generic parameters are made concrete as in the HLSL list: a floating-point or arithmetic ``T`` is ``float``, an
integer, atomic or unconstrained one ``int``, a logical one ``uint``, and a ``let`` size is 2. A function whose every
``[require(...)]`` names a stage-only capability is probed in that stage. Probes compile for SPIR-V, so a function
whose ``require`` names no SPIR-V target is rejected in its own probe and stays untested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import crossgl
from ..model import (
    OTHER_WIDTHS,
    VOID,
    Feature,
    Param,
    Signature,
    Type,
    raytracing_stage,
)
from .common import pick, split

# The scalar each generic constraint is probed with.
CONSTRAINTS = {
    "__BuiltinFloatingPointType": "float",
    "__BuiltinRealType": "float",
    "__BuiltinArithmeticType": "float",
    "__BuiltinSignedArithmeticType": "float",
    "__BuiltinType": "float",
    "IFloat": "float",
    "IArithmetic": "float",
    "IArithmeticAtomicable": "int",
    "IAtomicable": "int",
    "__BuiltinIntegerType": "int",
    "__BuiltinInt32Type": "int",
    "IInteger": "int",
    "__BuiltinLogicalType": "uint",
    "ILogical": "uint",
}
# ``[require(...)]`` capabilities only one kind of stage has, matched as prefixes of the capability atoms.
STAGE_CAPABILITIES = (
    ("fragment", "fragment"),
    ("getattributeatvertex", "fragment"),
    ("helper_lane", "fragment"),
    ("raytracing", "raytracing"),
    ("ser_", "raytracing"),
    ("meshshading", "mesh"),
)
_SCALARS = {
    "bool": "bool",
    "int": "int",
    "int32_t": "int",
    "uint": "uint",
    "uint32_t": "uint",
    "float": "float",
    "float32_t": "float",
    "double": "double",
    "float64_t": "double",
    "half": "half",
    "float16_t": "half",
    "int16_t": "int16",
    "uint16_t": "uint16",
    "int64_t": "int64",
    "uint64_t": "uint64",
    "int8_t4_packed": "int8x4_packed",
    "uint8_t4_packed": "uint8x4_packed",
}
_SHORTHAND = re.compile(
    r"(bool|int|uint|float|double|half|int32_t|uint32_t|float32_t|float64_t|float16_t|int16_t|uint16_t|int64_t"
    r"|uint64_t)([1-4])(?:x([1-4]))?"
)
# The targets a ``require`` capability atom such as ``cuda_glsl_hlsl_spirv`` is made of.
TARGETS = frozenset("cpp cuda glsl hlsl metal spirv wgsl llvm".split())
# Capability atoms that imply one target without naming it (source/slang/slang-capabilities.capdef).
_ATOM_TARGETS = {"optix_coopvec": "cuda", "optix_multilevel_traversal": "cuda"}
_GENERIC_TYPE = re.compile(r"(vector|matrix)\s*<(.*)>")
_QUALIFIERS = {
    "in": "in",
    "out": "out",
    "inout": "inout",
    "__ref": "inout",
    "__constref": "in",
    "const": "in",
    "__constexpr": "const",
    "constexpr": "const",
    "no_diff": "in",
    "uniform": "in",
}
_MODIFIERS = frozenset(
    "public static inline internal export __extern_cpp __unsafeForceInlineEarly __noinline nodiff".split()
)
_LEADING_CALL = re.compile(
    r"(__generic|__glsl_extension|__glsl_version|__spirv_version|__cuda_sm_version|__wgsl_extension|"
    r"__target_intrinsic|__intrinsic_op|__specialized_for_target)\s*(?=[(<])"
)
_HEAD = re.compile(
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
    head, attributes, generics = _strip_leading(" ".join(raw.split()))
    requires = " ".join(a for a in attributes if a.startswith("require"))
    # Splices may appear in the name, the types and the constraints, so the three are expanded together.
    variants = meta.expand("\0".join([head, requires, ", ".join(generics)]))
    result = []
    for variant in variants or []:
        head, requires, generic = variant.split("\0")
        match = _HEAD.fullmatch(head)
        if not match or match.group("result") in ("return", "typedef", "typealias"):
            continue
        name = match.group("name")
        if name.startswith("_") or "$" in name or name.startswith("operator"):
            continue
        inline = (match.group("generic") or "").strip("<> ")
        generic = ", ".join(g for g in (generic, inline) if g)
        result.append(Declaration(name, head, generic, requires))
    return result


def _generics(text: str) -> dict[str, str]:
    """Generic parameter name -> the scalar it is probed with, or ``"size"`` for a ``let`` size."""
    bound = {}
    for part in split(text):
        part = part.split("=")[0].strip()
        if not part:
            continue
        if part.startswith("let "):
            bound[part[4:].split(":")[0].strip()] = "size"
            continue
        name, _, constraint = part.partition(":")
        constraint = constraint.strip()
        bound[name.strip()] = (
            CONSTRAINTS.get(constraint, f"?{constraint}") if constraint else "int"
        )
    return bound


def slang_type(spelling: str, generics: dict[str, str]) -> Type:
    spelling = " ".join(spelling.split())
    if spelling == "void":
        return VOID
    if spelling in generics:
        scalar = generics[spelling]
        return (
            Type(scalar)
            if scalar in _SCALARS.values()
            else Type.other(scalar.lstrip("?") or spelling)
        )
    if spelling in _SCALARS:
        return Type(_SCALARS[spelling])
    shorthand = _SHORTHAND.fullmatch(spelling)
    if shorthand:
        scalar, rows, columns = shorthand.groups()
        if columns:
            return Type(_SCALARS[scalar], (int(columns), int(rows)))
        return (
            Type(_SCALARS[scalar])
            if rows == "1"
            else Type(_SCALARS[scalar], (int(rows),))
        )
    generic = _GENERIC_TYPE.fullmatch(spelling)
    if generic:
        parts = split(generic.group(2))
        element = slang_type(parts[0], generics)
        if element.opaque or element.shape:
            return Type.other(spelling)

        def size(text: str) -> int:
            text = text.strip()
            return int(text) if text.isdigit() else 2

        sizes = [size(p) for p in parts[1:]]
        if generic.group(1) == "vector" and len(sizes) == 1 and 2 <= sizes[0] <= 4:
            return Type(element.scalar, (sizes[0],))
        if generic.group(1) == "matrix" and len(sizes) >= 2:
            rows, columns = sizes[:2]
            if 2 <= rows <= 4 and 2 <= columns <= 4:
                return Type(element.scalar, (columns, rows))
    return Type.other(spelling)


def _parameter(text: str, generics: dict[str, str]) -> Param:
    text = re.sub(r"\[[^\]]*\]", "", text.split("=")[0]).strip()
    words = text.split()
    direction = "in"
    while words and words[0] in _QUALIFIERS:
        direction = _QUALIFIERS[words.pop(0)] if direction == "in" else direction
    spelling = " ".join(words[:-1]) if len(words) > 1 else " ".join(words)
    return Param(slang_type(spelling, generics), direction)


def signature(declaration: Declaration) -> Signature | None:
    match = _HEAD.fullmatch(declaration.head)
    if match is None:
        return None
    generics = _generics(declaration.generics)
    result = match.group("arrow") or match.group("result")
    params = [p for p in split(match.group("params")) if p.strip()]
    if any("..." in p for p in params):
        return None
    return Signature(
        slang_type(result, generics), tuple(_parameter(p, generics) for p in params)
    )


def stage(declaration: Declaration) -> str:
    """The stage a probe needs. ``require`` attributes are alternatives, so a stage is needed only when each of them
    names a stage-only capability."""
    stages = []
    for clause in re.findall(r"require\s*\(([^)]*)\)", declaration.requires):
        found = [
            name
            for atom in re.findall(r"\w+", clause)
            for prefix, name in STAGE_CAPABILITIES
            if atom.startswith(prefix)
        ]
        if not found:
            return "compute"
        stages.append(found[0])
    return stages[0] if stages else "compute"


def _stage(name: str, declaration: Declaration) -> str:
    """:func:`stage`, with a ray-tracing function probed in the ray-tracing stage that has it, and DispatchMesh, which
    launches mesh workgroups, in the task stage."""
    found = stage(declaration)
    if found == "mesh" and name == "DispatchMesh":
        return "task"
    return raytracing_stage(name) if found == "raytracing" else found


def _preference(signature: Signature) -> tuple[bool, bool, int]:
    """Sort key among overloads: value-returning before ``void``, 32-bit before 64-bit or 16-bit, and vectors
    before scalars before matrices, so a generic function is probed with two-component vectors as in the HLSL list.
    """
    kinds = [signature.result, *(p.type for p in signature.params)]
    shapes = {len(k.shape) for k in kinds if not k.opaque}
    rank = 2 if 2 in shapes else 0 if 1 in shapes else 1
    return (
        signature.result == VOID
        and any(p.direction == "out" for p in signature.params),
        any(k.scalar in OTHER_WIDTHS for k in kinds),
        rank,
    )


def spirv_target(declaration: Declaration) -> bool:
    """Whether some ``require`` alternative allows SPIR-V; one that names no target allows every target."""
    clauses = re.findall(r"require\s*\(([^)]*)\)", declaration.requires)
    for clause in clauses or [""]:
        atoms = [_ATOM_TARGETS.get(atom, atom) for atom in re.findall(r"\w+", clause)]
        targets = {
            part
            for atom in atoms
            if set(atom.split("_")) <= TARGETS
            for part in atom.split("_")
        }
        if not targets or "spirv" in targets:
            return True
    return False


def _spell(kind: Type) -> str:
    if len(kind.shape) == 2:
        columns, rows = kind.shape
        return f"{kind.scalar}{rows}x{columns}"
    return f"{kind.scalar}{kind.shape[0]}" if kind.shape else kind.scalar


def explicit(declaration: Declaration, chosen: Signature) -> str:
    """How the native probe calls a function whose result type is a generic parameter no argument fixes, as
    ``bit_cast<int2>``; an empty string when the arguments fix every generic parameter.
    """
    match = _HEAD.fullmatch(declaration.head)
    generics = _generics(declaration.generics)
    if not match or chosen.result.opaque:
        return ""
    params = match.group("params")
    free = [
        name
        for name, bound in generics.items()
        if bound != "size" and not re.search(rf"\b{re.escape(name)}\b", params)
    ]
    result = match.group("arrow") or match.group("result")
    if free and re.search(rf"\b{re.escape(free[0])}\b", result):
        return f"{declaration.name}<{_spell(chosen.result)}>"
    return ""


def parse(
    core: str, hlsl: str, glsl_names: set[str] | frozenset[str] = frozenset()
) -> list[Feature]:
    overloads: dict[str, list[tuple[str, Declaration]]] = {}
    for group, text in (("core", core), ("hlsl", hlsl)):
        for declaration in declarations(text):
            overloads.setdefault(declaration.name, []).append((group, declaration))
    features = []
    for name, declared in sorted(overloads.items()):
        candidates = [(s, d) for s, d in ((signature(d), d) for _, d in declared) if s]
        note = "" if candidates else "variadic"
        if not any(spirv_target(d) for _, d in declared):
            candidates, note = [], "needs a target other than SPIR-V"
        chosen = None
        if candidates:
            chosen = pick(sorted((s for s, _ in candidates), key=_preference))
        source = next((d for s, d in candidates if s == chosen), declared[0][1])
        features.append(
            Feature(
                "slang",
                "function",
                name,
                chosen,
                group=declared[0][0],
                crossgl=crossgl.spelling("slang", name, glsl_names),
                stage=_stage(name, declared[0][1]),
                note=note,
                spelling=explicit(source, chosen) if chosen else "",
            )
        )
    return features

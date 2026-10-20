"""GLSL built-in functions from the GLSL 4.60 specification source (KhronosGroup/GLSL).

Every ``[source,glsl]`` block of ``chapters/builtinfunctions.adoc`` lists prototypes. Desktop GLSL text is kept and
``ifdef::ESSL[]`` text dropped. Sections that exist only for the compatibility profile or are deprecated are left
out, because a core-profile shader cannot use them.
"""

from __future__ import annotations

import re

from ..model import Feature, Signature, Type
from .common import glsl_type, pick, signature, slug

LEFT_OUT = {
    "compatibility-profile-texture-functions": "compatibility profile only",
    "noise-functions": "deprecated since GLSL 4.40",
    "ftransform": "compatibility profile only",
}
STAGES = {
    "geometry-shader-functions": "geometry",
    "fragment-processing-functions": "fragment",
}
# The generic type names of section 8, made concrete for a probe.
GENERIC = {
    "genFType": "vec3",
    "genDType": "dvec3",
    "genIType": "ivec3",
    "genUType": "uvec3",
    "genBType": "bvec3",
    "vec": "vec3",
    "ivec": "ivec3",
    "uvec": "uvec3",
    "bvec": "bvec3",
    "mat": "mat3",
}
_PRECISION = re.compile(r"\b(?:highp|mediump|lowp)\s+")
_OPTIONAL = re.compile(r"\[[^\[\]]*\]")
_PROTOTYPE = re.compile(r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(([^(){};]*)\)")


def _type(spelling: str) -> Type:
    return glsl_type(GENERIC.get(spelling, spelling))


def _param(text: str) -> tuple[Type, str]:
    words = text.split()
    direction = "in"
    if words and words[0] in ("in", "out", "inout"):
        direction = words.pop(0)
    while words and words[0] in (
        "const",
        "coherent",
        "volatile",
        "restrict",
        "readonly",
        "writeonly",
    ):
        words.pop(0)
    if not words:
        raise ValueError(f"empty parameter {text!r}")
    return _type(words[0]), direction


def parse_prototype(text: str) -> tuple[str, Signature] | None:
    match = _PROTOTYPE.search(_OPTIONAL.sub("", _PRECISION.sub("", text)))
    if not match:
        return None
    result, name, params = match.groups()
    if result in ("return", "if", "else"):
        return None
    parts = [p.strip() for p in params.split(",") if p.strip() and p.strip() != "void"]
    return name, signature(_type(result), [_param(p) for p in parts])


def _blocks(text: str):
    """Yield ``(section, subsection, block)`` for every desktop-GLSL source block."""
    section = subsection = ""
    essl: list[bool] = (
        []
    )  # one entry per open ifdef/ifndef: whether it holds ES-only text
    in_block = False
    body: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        directive = re.fullmatch(r"(ifdef|ifndef|endif)::(\w+)\[\]", stripped)
        if directive:
            kind, name = directive.groups()
            if kind == "endif":
                if essl:
                    essl.pop()
            else:
                essl.append((name == "ESSL") == (kind == "ifdef"))
            continue
        if any(essl):
            continue
        if in_block:
            if stripped == "----":
                in_block = False
                yield section, subsection, "\n".join(body)
            else:
                body.append(line)
            continue
        heading = re.match(r"(=+)\s+(.*)", line)
        if heading:
            level, title = len(heading.group(1)), slug(heading.group(2))
            if level == 2:
                section, subsection = title, ""
            elif level == 3:
                subsection = title
        elif (
            stripped == "----" and index and lines[index - 1].strip() == "[source,glsl]"
        ):
            in_block, body = True, []


def overloads(text: str) -> dict[str, tuple[str, list[Signature]]]:
    """Every desktop-GLSL prototype, by function name: ``{name: (section, [signature, ...])}``."""
    found: dict[str, tuple[str, list[Signature]]] = {}
    for section, subsection, block in _blocks(text):
        if "{" in block or section in LEFT_OUT or subsection in LEFT_OUT:
            continue
        for statement in re.split(r"\n(?=\S)", block):
            parsed = parse_prototype(" ".join(statement.split()))
            if parsed:
                name, sig = parsed
                found.setdefault(name, (section, []))[1].append(sig)
    return found


def parse(text: str) -> list[Feature]:
    return [
        Feature(
            "glsl",
            "function",
            name,
            pick(signatures),
            group=group,
            crossgl=name,
            stage=STAGES.get(group, "compute"),
        )
        for name, (group, signatures) in sorted(overloads(text).items())
    ]

"""GLSL built-in functions from the GLSL 4.60 specification source (KhronosGroup/GLSL).

Every ``[source,glsl]`` block of ``chapters/builtinfunctions.adoc`` lists prototypes. Desktop GLSL text is kept and
``ifdef::ESSL[]`` text dropped. Sections that exist only for the compatibility profile or are deprecated are left
out, because a core-profile shader cannot use them.
"""

from __future__ import annotations

import re

from ..model import Feature, Signature, Type
from ..objects import UNIFORMS
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
# Texture functions that compute an implicit level of detail, from derivatives only a fragment shader has.
IMPLICIT_LOD = frozenset(
    {"texture", "textureOffset", "textureProj", "textureProjOffset", "textureQueryLod"}
)
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
    "gvec4": "vec4",
}
# ``gsampler2D`` and ``gimage2D`` stand for the float, int and uint forms; a probe uses the float one.
_GENERIC_OPAQUE = re.compile(r"g((?:sampler|image|subpassInput)\w*)")
# Section 8.12 writes image functions with this placeholder for an image and its coordinate.
_IMAGE_PARAMS = "IMAGE_PARAMS"
_PRECISION = re.compile(r"\b(?:highp|mediump|lowp)\s+")
# Optional trailing arguments, written ``[, float bias]``; ``offsets[4]`` is an array parameter instead.
_OPTIONAL = re.compile(r"\[\s*,[^\[\]]*\]")
_PROTOTYPE = re.compile(r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(([^(){};]*)\)")


def concrete_type(spelling: str) -> Type:
    """The probe's type for a GLSL type spelling, with the generic names of section 8 made concrete."""
    return glsl_type(GENERIC.get(spelling, _GENERIC_OPAQUE.sub(r"\1", spelling)))


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
    array = re.fullmatch(r"\w+\[(\d+)\]", words[-1])
    if array:
        # textureGatherOffsets' ``ivec2 offsets[4]``, whose offsets must be constant expressions (GLSL 4.60,
        # section 8.9.4).
        kind = concrete_type(words[0])
        return Type(kind.scalar, kind.shape, count=int(array.group(1))), "const"
    if "[" in words[-1]:
        return Type.other(words[0] + words[-1][words[-1].index("[") :]), direction
    kind = concrete_type(words[0])
    if words[-1] == "offset" and kind.scalar == "int":
        # A texel offset must be a constant expression (GLSL 4.60, section 8.9.2).
        direction = "const"
    if words[-1] == "stream":
        # So must a vertex stream (GLSL 4.60, section 8.16).
        direction = "const"
    return kind, direction


def parse_prototype(text: str) -> tuple[str, Signature] | None:
    match = _PROTOTYPE.search(_OPTIONAL.sub("", _PRECISION.sub("", text)))
    if not match:
        return None
    result, name, params = match.groups()
    if result in ("return", "if", "else"):
        return None
    # The 2D form of the placeholder; an atomic's image holds the scalar type the function returns.
    image = {"int": "iimage2D", "uint": "uimage2D"}.get(result, "image2D")
    params = params.replace(_IMAGE_PARAMS, f"{image} image, ivec2 P")
    parts = [p.strip() for p in params.split(",") if p.strip() and p.strip() != "void"]
    return name, signature(concrete_type(result), [_param(p) for p in parts])


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


def _stage(name: str, group: str) -> str:
    return "fragment" if name in IMPLICIT_LOD else STAGES.get(group, "compute")


def parse(text: str) -> list[Feature]:
    return [
        Feature(
            "glsl",
            "function",
            name,
            pick(signatures, UNIFORMS["glsl"]),
            group=group,
            crossgl=name,
            stage=_stage(name, group),
        )
        for name, (group, signatures) in sorted(overloads(text).items())
    ]

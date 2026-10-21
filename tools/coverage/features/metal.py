"""Metal standard-library functions, listed from the headers of Apple's Metal toolchain.

Apple publishes the Metal Shading Language specification only as a PDF, so the list is generated once, on a Mac
with the Metal toolchain, by ``python -m tools.coverage.features.metal``, which reads the toolchain's standard-library
headers and rewrites ``metal.txt`` next to this module. The file records the toolchain and language version it came
from; regenerate it only to move to a newer version, and say so in the pull request, as when moving an upstream pin.

Each line of ``metal.txt`` is ``header name result(param, ...)``: one overload per function, chosen as the GLSL list
chooses its generic types, with template parameters made concrete. ``inout`` marks the atomic object an atomic
function updates, and ``memory_order`` and ``mem_flags`` an enumeration the probe passes as a constant. A function
with no overload a probe can call is listed with ``-`` in place of the signature.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from .. import spellings
from ..model import VOID, Feature, Param, Signature, Type
from .common import prototype, split

LIST = Path(__file__).with_name("metal.txt")
HEADERS = (
    "metal_math",
    "metal_common",
    "metal_integer",
    "metal_relational",
    "metal_geometric",
    "metal_matrix",
    "metal_pack",
    "metal_graphics",
    "metal_compute",
    "metal_atomic",
    "metal_simdgroup",
    "metal_quadgroup",
)
STAGES = {"metal_graphics": "fragment"}
_SCALARS = {
    "float": "float",
    "int": "int",
    "uint": "uint",
    "bool": "bool",
    "long": "int64",
    "ulong": "uint64",
    "half": "half",
}
# Narrow integer parameters (lane ids, deltas) take a 32-bit argument, which converts implicitly.
_WIDENED = {"ushort": "uint", "short": "int"}
_QUALIFIERS = re.compile(
    r"\b(?:const|constexpr|static|inline|thread|device|threadgroup|constant|volatile|explicit|METAL_\w+)\b"
    r"|\bcoherent\s*\(\s*\w+\s*\)"
)
_MACRO = re.compile(r"\bMETAL_\w+\b")
_COHERENT = re.compile(r"\bcoherent\s*\(\s*\w+\s*\)")
_DECLARATION = re.compile(
    r"METAL_FUNC\s+(?P<result>.*?)\b(?P<name>[A-Za-z_]\w*)\s*\(", re.S
)
# Enumerations a probe passes as a constant; :class:`~tools.coverage.languages.Metal` names the enumerator.
ENUMS = ("memory_order", "mem_flags")
# The masks a ballot returns. A probe reads one as its ``vote_t`` integer, which Metal converts it to explicitly.
VOTES = ("simd_vote", "quad_vote")
_ATOMIC = re.compile(r"_atomic\s*<\s*T\b[^>]*>")


def _template_type(template: str) -> str:
    """The type a probe uses for a template parameter ``T``, from the declaration's ``enable_if`` constraint."""
    if re.search(r"integral|is_integer", template):
        return "int"
    if "bool" in template and "floating" not in template:
        return "bool"
    return "float"


def _atomic_type(result: str) -> str:
    """The type an atomic function's ``T`` and ``U`` take: int, except for the 64-bit atomic max and min, which Metal
    declares only for ulong."""
    return "ulong" if re.search(r"_valid_m(?:ax|in)_type", result) else "int"


def _concrete(spelling: str, template: str, bound: str = "") -> str:
    """The probe type for one spelling; ``bound`` is the type of ``T`` and ``U`` when the header fixes it."""
    spelling = _QUALIFIERS.sub("", spelling).strip()
    enabled = re.fullmatch(r"enable_if_t\s*<(.*)>", spelling, re.S)
    if enabled:
        # enable_if_t<condition, type> is the type, and enable_if_t<condition> is void.
        arguments = split(enabled.group(1))
        if len(arguments) <= 2:
            spelling = arguments[-1] if len(arguments) == 2 else "void"
    if spelling in ("T", "U") and bound:
        return bound
    if spelling == "T" and template:
        return _template_type(template)
    matrix = re.fullmatch(r"matrix\s*<\s*T\s*,\s*(\w+)\s*,\s*(\w+)\s*>", spelling)
    if matrix:
        columns, rows = (n if n.isdigit() else "3" for n in matrix.groups())
        return f"float{columns}x{rows}"
    return _WIDENED.get(spelling, spelling)


def _result(spelling: str) -> bool:
    """Whether a probe can receive ``spelling``: a value, or a ballot's mask."""
    return spelling in VOTES or _value(spelling)


def _value(spelling: str) -> bool:
    """Whether a probe can pass or receive ``spelling``: a value, an atomic operand in the output buffer, or an
    enumeration it passes as a constant."""
    spelling = re.sub(r"^(?:out|inout|address) ", "", spelling)
    return spelling in ENUMS or bool(
        re.fullmatch(
            r"(float|int|uint|bool|long|ulong|half)([2-4])?|(?:float|half)[2-4]x[2-4]|void",
            spelling,
        )
    )


def _arguments(declaration: str, start: int) -> str:
    """The text between the parenthesis at ``start`` and the one that closes it."""
    depth = 0
    for index in range(start, len(declaration)):
        depth += {"(": 1, ")": -1}.get(declaration[index], 0)
        if depth == 0:
            return declaration[start + 1 : index]
    return declaration[start + 1 :]


def _declared(part: str, template: str, bound: str) -> str | None:
    """The listed type of one declared parameter, or ``None`` for one with a default, which the probe leaves out."""
    if "=" in part:
        return None
    # Attribute macros such as METAL_MAYBE_UNDEF follow the parameter name.
    part = _MACRO.sub("", part).strip()
    if "*" in part:
        # A pointer to an atomic object is the probe's output buffer element; no other pointer is probed.
        pointee = _QUALIFIERS.sub("", part.rsplit("*", 1)[0]).strip()
        if bound and _ATOMIC.fullmatch(pointee):
            return f"inout {bound}"
        kind = _concrete(pointee, template, bound)
        # A pointer to a thread-local value, such as compare-exchange's expected value, takes a local's address.
        if part.startswith("thread ") and _value(kind):
            return f"address {kind}"
        return f"{kind}*"
    reference = "&" in part
    words = part.replace("&", " ").rsplit(None, 1)
    kind = _concrete(words[0] if len(words) == 2 else part, template, bound)
    return f"out {kind}" if reference else kind


def declarations(header: str, text: str):
    """Yield ``(name, result, params)`` for every namespace-scope function declared in one header."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    lines, index, template = text.splitlines(), 0, ""
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.startswith("template"):
            template = line
            while template.count("<") > template.count(">") and index < len(lines):
                template += " " + lines[index].strip()
                index += 1
            continue
        if not line.startswith("METAL_FUNC"):
            if line.strip():
                template = ""
            continue
        declaration = line
        while declaration.count("(") > declaration.count(")") or "(" not in declaration:
            if index >= len(lines):
                break
            declaration += " " + lines[index].strip()
            index += 1
        # coherent(device) qualifies a type: its parenthesis is not the parameter list's.
        declaration = _COHERENT.sub("", declaration)
        match = _DECLARATION.match(declaration)
        # An operator's symbol stops the name match, which would then take a later word before a parenthesis for the
        # function's name. Helpers start with an underscore.
        if (
            not match
            or match.group("name").startswith(("_", "operator"))
            or re.search(r"\boperator\b", match.group("result"))
        ):
            template = ""
            continue
        # Metal's atomic functions fix the types their template parameters may take; see _atomic_type.
        bound = _atomic_type(match.group("result")) if header == "metal_atomic" else ""
        params = []
        for part in split(_arguments(declaration, match.end() - 1)):
            kind = _declared(part, template, bound)
            if kind is not None:
                params.append(kind)
        yield (
            match.group("name"),
            _concrete(match.group("result"), template, bound),
            params,
        )
        template = ""


_PREFERRED = ("float", "int", "uint", "bool", "long", "ulong", "half")


def _preference(overload: tuple[str, list[str]]) -> tuple:
    result, params = overload
    first = params[0] if params else result
    scalar = re.match(r"[a-z]+", first).group(0) if re.match(r"[a-z]+", first) else ""
    width = re.search(r"(\d)$", first)
    return (
        not (_result(result) and all(_value(k) for k in params)),
        _PREFERRED.index(scalar) if scalar in _PREFERRED else len(_PREFERRED),
        {None: 1, "3": 0}.get(width and width.group(1), 2),
        # Of two otherwise equal overloads, such as an atomic function with and without memory flags, the shorter.
        len(params),
    )


def generate(include: Path, toolchain: str, language: str) -> str:
    """The text of ``metal.txt`` for the headers under ``include``."""
    lines = [
        "# Generated by python -m tools.coverage.features.metal; do not edit.",
        f"# toolchain: {toolchain}",
        f"# language: Metal Shading Language {language}",
    ]
    listed: set[str] = set()
    for header in HEADERS:
        overloads: dict[str, list[tuple[str, list[str]]]] = {}
        for name, result, params in declarations(
            header, (include / header).read_text()
        ):
            overloads.setdefault(name, []).append((result, params))
        # A function declared in several headers, such as abs for floats and for integers, is listed once.
        for name, found in sorted(overloads.items()):
            if name in listed:
                continue
            listed.add(name)
            result, params = min(found, key=_preference)
            usable = _result(result) and all(_value(k) for k in params)
            lines.append(
                f"{header} {name} "
                + (f"{result}({', '.join(params)})" if usable else "-")
            )
    return "\n".join(lines) + "\n"


def source(text: str) -> dict[str, str]:
    """Where ``metal.txt`` came from, read from its header lines, in the shape of the report's other sources."""
    found = dict(re.findall(r"^# (toolchain|language): (.+)$", text, re.M))
    return {
        "path": f"tools/coverage/features/{LIST.name}",
        "describes": (
            f"{found.get('language', 'Metal Shading Language')} standard library, "
            f"from {found.get('toolchain', 'an unknown toolchain')}"
        ),
    }


def metal_type(spelling: str) -> Type:
    if spelling == "void":
        return VOID
    match = re.fullmatch(
        r"(float|int|uint|bool|long|ulong|half)(?:([2-4])(?:x([2-4]))?)?", spelling
    )
    if not match:
        return Type.other(spelling)
    scalar, first, second = match.groups()
    if second:
        return Type(_SCALARS[scalar], (int(first), int(second)))
    return Type(_SCALARS[scalar], (int(first),) if first else ())


def _param(text: str) -> Param:
    text = text.strip()
    direction, _, kind = text.rpartition(" ")
    if text in ENUMS:
        return Param(Type.other(text), "const")
    return Param(metal_type(kind), direction or "in")


def parse(
    text: str, glsl_names: set[str] | frozenset[str] = frozenset()
) -> list[Feature]:
    features = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        header, name, declared = line.split(" ", 2)
        signature = None
        if declared != "-":
            result, params = re.fullmatch(r"(\w+)\((.*)\)", declared).groups()
            signature = Signature(
                metal_type(result),
                tuple(_param(p) for p in params.split(",") if p.strip()),
            )
        spelling, counterpart = spellings.spelling("metal", name, glsl_names), None
        note = "" if signature else "no overload takes only value types"
        if name in spellings.COUNTERPARTS["metal"]:
            spelling, counterpart = prototype(spellings.COUNTERPARTS["metal"][name])
        elif signature and any(p.type.opaque in ENUMS for p in signature.params):
            spelling = ""
            note = "CrossGL has no operation that takes a memory order or memory flags"
        features.append(
            Feature(
                "metal",
                "function",
                name,
                signature,
                group=header.removeprefix("metal_"),
                crossgl=spelling,
                stage=STAGES.get(header, "compute"),
                note=note,
                counterpart=counterpart,
            )
        )
    return features


def _toolchain() -> tuple[Path, str, str]:
    """The standard-library include directory, toolchain version and language version of the installed toolchain."""
    metal = ("xcrun", "-sdk", "macosx", "metal")
    compiler = Path(
        subprocess.run(
            [*metal[:3], "-f", "metal"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    found = sorted(compiler.parent.parent.glob("metal/*/lib/clang/*/include/metal"))
    if not found:
        raise SystemExit(f"no Metal standard-library headers next to {compiler}")
    version = subprocess.run(
        [*metal, "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    probe = subprocess.run(
        [*metal, "-x", "metal", "-E", "-P", "-"],
        input="__METAL_VERSION__\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[-1]
    return found[-1], version, f"{int(probe) // 100}.{int(probe) % 100 // 10}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.coverage.features.metal", description=__doc__
    )
    parser.add_argument("--output", type=Path, default=LIST)
    args = parser.parse_args(argv)
    include, toolchain, language = _toolchain()
    args.output.write_text(generate(include, toolchain, language))
    print(f"wrote {args.output} from {toolchain}, MSL {language}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Metal standard-library functions, listed from the headers of Apple's Metal toolchain.

Apple publishes the Metal Shading Language specification only as a PDF, so the list is generated once, on a Mac
with the Metal toolchain, by ``python -m tools.coverage.features.metal``, which reads the toolchain's standard-library
headers and rewrites ``metal.txt`` next to this module. The file records the toolchain and language version it came
from; regenerate it only to move to a newer version, and say so in the pull request, as when moving an upstream pin.

Each line of ``metal.txt`` is ``header name result(param, ...)``: one overload per function, chosen as the GLSL list
chooses its generic types, with template parameters made concrete. A function with no overload a probe can call is
listed with ``-`` in place of the signature.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from .. import crossgl
from ..model import VOID, Feature, Param, Signature, Type
from .common import split

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
_SCALARS = {"float": "float", "int": "int", "uint": "uint", "bool": "bool"}
# Narrow integer parameters (lane ids, deltas) take a 32-bit argument, which converts implicitly.
_WIDENED = {"ushort": "uint", "short": "int"}
_QUALIFIERS = re.compile(
    r"\b(?:const|constexpr|static|inline|thread|device|threadgroup|constant|volatile|explicit|METAL_\w+)\b"
)
_DECLARATION = re.compile(
    r"METAL_FUNC\s+(?P<result>.*?)\b(?P<name>[A-Za-z]\w*)\s*\((?P<params>.*)\)", re.S
)


def _template_type(template: str) -> str:
    """The type a probe uses for a template parameter ``T``, from the declaration's ``enable_if`` constraint."""
    if re.search(r"integral|is_integer", template):
        return "int"
    if "bool" in template and "floating" not in template:
        return "bool"
    return "float"


def _concrete(spelling: str, template: str) -> str:
    spelling = _QUALIFIERS.sub("", spelling).strip()
    enabled = re.fullmatch(r"enable_if_t\s*<.*,\s*(\w+)\s*>", spelling, re.S)
    if enabled:
        spelling = enabled.group(1)
    if spelling == "T" and template:
        return _template_type(template)
    matrix = re.fullmatch(r"matrix\s*<\s*T\s*,\s*(\w+)\s*,\s*(\w+)\s*>", spelling)
    if matrix:
        columns, rows = (n if n.isdigit() else "3" for n in matrix.groups())
        return f"float{columns}x{rows}"
    return _WIDENED.get(spelling, spelling)


def _value(spelling: str) -> bool:
    spelling = spelling.removeprefix("out ")
    return bool(
        re.fullmatch(r"(float|int|uint|bool)([2-4])?|float[2-4]x[2-4]|void", spelling)
    )


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
        match = _DECLARATION.match(declaration)
        if not match or match.group("name").startswith(("_", "operator")):
            template = ""
            continue
        params = []
        for part in split(match.group("params")):
            part = re.sub(r"=.*$", "", part).strip()
            reference = "&" in part
            words = part.replace("&", " ").rsplit(None, 1)
            kind = _concrete(words[0] if len(words) == 2 else part, template)
            params.append(f"out {kind}" if reference else kind)
        yield match.group("name"), _concrete(match.group("result"), template), params
        template = ""


def _preference(overload: tuple[str, list[str]]) -> tuple:
    result, params = overload
    kinds = [result, *params]
    first = params[0] if params else result
    scalar = re.match(r"[a-z]+", first).group(0) if re.match(r"[a-z]+", first) else ""
    width = re.search(r"(\d)$", first)
    return (
        not all(_value(k) for k in kinds),
        ("float", "int", "uint", "bool").index(scalar) if scalar in _SCALARS else 4,
        {None: 1, "3": 0}.get(width and width.group(1), 2),
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
            usable = all(_value(k) for k in [result, *params])
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
    match = re.fullmatch(r"(float|int|uint|bool)(?:([2-4])(?:x([2-4]))?)?", spelling)
    if not match:
        return Type.other(spelling)
    scalar, first, second = match.groups()
    if second:
        return Type(_SCALARS[scalar], (int(first), int(second)))
    return Type(_SCALARS[scalar], (int(first),) if first else ())


def _param(text: str) -> Param:
    text = text.strip()
    if text.startswith("out "):
        return Param(metal_type(text.removeprefix("out ")), "out")
    return Param(metal_type(text))


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
        features.append(
            Feature(
                "metal",
                "function",
                name,
                signature,
                group=header.removeprefix("metal_"),
                crossgl=crossgl.spelling("metal", name, glsl_names),
                stage=STAGES.get(header, "compute"),
                note="" if signature else "no overload takes only value types",
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

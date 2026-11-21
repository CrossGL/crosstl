"""The template type, and what the templates of every language share."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    body: str = ""
    declarations: str = ""
    prelude: str = ""
    params: str = ""
    program: str = ""
    entry: str = ""
    output: str = ""
    # The compiler target the probe is checked for when it is not the language's usual one (Feature.target).
    target: str = ""


ABSENT = Template(entry="<absent>")

# What only the graphics APIs have: fragment discard and every stage but compute.
GRAPHICS_ONLY = (
    "discard",
    "vertex",
    "fragment",
    "geometry",
    "tessellation-control",
    "tessellation-evaluation",
    "task",
    "mesh",
    "ray-generation",
    "intersection",
    "any-hit",
    "closest-hit",
    "miss",
    "callable",
)


def scalar(spelling: str, source: str = "iin", prelude: str = "") -> Template:
    return Template(
        f"{spelling} v = {spelling}({source}[i]);\noutp[i] = float(v * v);",
        prelude=prelude,
    )


def narrow_scalar(spelling: str, prelude: str = "") -> Template:
    """The probe of an 8- or 16-bit integer type in a language that promotes it to ``int`` in arithmetic.

    C and C++ convert 8- and 16-bit operands to ``int`` before they multiply them (C11 6.3.1.1, C++ [conv.prom]),
    and so do MSL, which follows C++ here, and CrossGL. GLSL, Slang and SPIR-V multiply at the operands' width. So
    the probe converts the product back to the operands' type, and every language computes the same wrapped product,
    -87 for 13 as a signed 8-bit integer. C leaves that conversion to the implementation (C11 6.3.1.3), and C++20
    defines it as keeping the low bits ([conv.integral]), as SPIR-V's ``OpSConvert`` does.
    """
    return Template(
        f"{spelling} v = {spelling}(iin[i]);\noutp[i] = float({spelling}(v * v));",
        prelude=prelude,
    )

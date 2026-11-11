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

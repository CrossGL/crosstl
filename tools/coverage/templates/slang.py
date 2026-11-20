"""Probe templates for Slang, which compiles most HLSL as it is."""

from __future__ import annotations

import re
from dataclasses import replace

from . import (
    abstractions,
    bindings,
    draws,
    extensions,
    fragments,
    interfaces,
    interlocks,
    memory,
    meshes,
    pipeline,
    pointers,
    primitives,
    qualifiers,
    rasterization,
    rays,
    specializations,
    subgroups,
    tasks,
    tessellation,
    vectors,
    vertices,
)
from .base import ABSENT, Template, scalar
from .hlsl import HLSL, raytracing_stages


def _slang(template: Template, stage: str = "") -> Template:
    """An HLSL template as Slang: slangc assigns bindings itself, and a stage program names its stage."""
    if template is ABSENT:
        return template
    declarations = re.sub(r"\s*:\s*register\(\w+\)", "", template.declarations)
    program = f'[shader("{stage}")]\n{template.program}' if stage else template.program
    return replace(template, declarations=declarations, program=program)


# Slang compiles HLSL, adds 8-bit integers, and reads the workgroup count through WorkgroupCount().
SLANG = {
    **{name: _slang(template) for name, template in HLSL.items()},
    "int8": scalar("int8_t"),
    "uint8": scalar("uint8_t"),
    "num-workgroups": Template("outp[i] = float(WorkgroupCount().x);"),
    **abstractions.SLANG,
    **{name: _slang(template) for name, template in vectors.SLANG.items()},
    **fragments.SLANG,
    **interfaces.SLANG,
    **vertices.SLANG,
    **draws.SLANG,
    **rasterization.SLANG,
    **pipeline.SLANG,
    **extensions.SLANG,
    **primitives.SLANG,
    **tessellation.SLANG,
    **meshes.SLANG,
    **tasks.SLANG,
    **rays.SLANG,
    **bindings.SLANG,
    **interlocks.SLANG,
    **subgroups.SLANG,
    **specializations.SLANG,
    **qualifiers.SLANG,
    **pointers.SLANG,
    **memory.ELSEWHERE,
    "vertex": _slang(HLSL["vertex"], "vertex"),
    "fragment": _slang(HLSL["fragment"], "fragment"),
    # The stage attribute goes on the entry point, after the output struct.
    "geometry": replace(
        HLSL["geometry"],
        program=HLSL["geometry"].program.replace(
            "[maxvertexcount", '[shader("geometry")]\n[maxvertexcount'
        ),
    ),
    "discard": _slang(HLSL["discard"], "fragment"),
    # The stage attribute goes on the entry point, ahead of its other attributes.
    **{
        name: replace(
            HLSL[name],
            program=HLSL[name].program.replace(
                attribute, f'[shader("{stage}")]\n{attribute}', 1
            ),
        )
        for name, stage, attribute in (
            ("tessellation-control", "hull", "[domain("),
            ("tessellation-evaluation", "domain", "[domain("),
            ("mesh", "mesh", "[numthreads("),
            ("task", "amplification", "[numthreads("),
        )
    },
    # slangc takes no payload access qualifiers.
    **{
        name: replace(template, program=template.program.replace(" : register(u0)", ""))
        for name, template in raytracing_stages(
            "struct Payload {\n    float4 color;\n};\n"
        ).items()
    },
}

"""Geometry stages that write several vertex streams, in SPIR-V.

GLSL's ``EmitStreamVertex(n)`` and ``EndStreamPrimitive(n)``, with each output qualified ``layout(stream = n)``
(GLSL 4.60, geometry outputs and geometry shader functions), are CrossGL's ``EmitStreamVertex(n)``, ``EndStreamPrimitive(n)`` and
``@stream(n)``. SPIR-V writes them as ``OpEmitStreamVertex`` and ``OpEndStreamPrimitive``, whose stream operand is a
constant, with the ``GeometryStreams`` capability and a ``Stream`` decoration on each output (SPIR-V 1.6, the
``Stream`` decoration and the primitive instructions). The SPIR-V emitter refused both calls.

:func:`geometry_output_streams` reads each output's ``@stream(n)``; :func:`stream_decorations` decorates every
``Output`` variable of a module that emits to a stream with its stream, 0 when it names none, as glslang does.
"""

from __future__ import annotations

import re

__all__ = [
    "STREAM_FUNCTIONS",
    "geometry_output_streams",
    "stream_decorations",
]

STREAM_FUNCTIONS = {
    "EmitStreamVertex": "OpEmitStreamVertex",
    "EndStreamPrimitive": "OpEndStreamPrimitive",
}
_OUTPUT = re.compile(r"^\s*(%\w+) = OpVariable %\w+ Output\b")
_NAME = re.compile(r'^\s*OpName (%\w+) "([^"]*)"')


def _stream_of(node):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() != "stream":
            continue
        arguments = getattr(attribute, "arguments", None) or []
        if arguments:
            value = getattr(arguments[0], "value", arguments[0])
            try:
                return int(str(value).strip().rstrip("uU"), 0)
            except ValueError:
                return None
    return None


def geometry_output_streams(ast):
    """``{name: stream}`` for every declaration of ``ast`` that carries ``@stream(n)``."""
    streams = {}
    walk = getattr(ast, "walk", None)
    for node in walk() if callable(walk) else []:
        name = getattr(node, "name", None)
        if not isinstance(name, str):
            continue
        stream = _stream_of(node)
        if stream is not None:
            streams.setdefault(name, stream)
    return streams


def stream_decorations(global_variables, debug_names, streams):
    """``(decorations, unplaced)``: a ``Stream`` decoration per ``Output`` variable, and the stream outputs not found.

    ``global_variables`` and ``debug_names`` are the module's ``OpVariable`` and ``OpName`` lines. An output whose
    name ``streams`` does not list writes stream 0.
    """
    names = {}
    for line in debug_names:
        match = _NAME.match(line)
        if match:
            names.setdefault(match.group(1), match.group(2))
    decorations, placed = [], set()
    for line in global_variables:
        match = _OUTPUT.match(line)
        if not match:
            continue
        name = names.get(match.group(1))
        stream = streams.get(name, 0)
        if name in streams:
            placed.add(name)
        decorations.append(f"OpDecorate {match.group(1)} Stream {stream}")
    unplaced = sorted(
        name for name, stream in streams.items() if stream and name not in placed
    )
    return decorations, unplaced

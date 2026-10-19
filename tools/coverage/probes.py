"""Minimal probe programs: one call of one feature inside a compute kernel.

Arguments are read from input buffers, so no call folds to a constant, and the result is written to an output
buffer, so no call is dead code. Output probes are CrossGL; input probes are written in the source language.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Feature, Signature, Type

PROBE_LANGUAGES = ("crossgl", "glsl", "hlsl", "metal")
SUFFIX = {"crossgl": ".cgl", "glsl": ".comp", "hlsl": ".hlsl", "metal": ".metal"}
_BUFFERS = {"float": "fin", "int": "iin", "uint": "uin"}


class ProbeError(ValueError):
    """The feature's signature cannot be expressed as a probe yet."""


@dataclass(frozen=True)
class Probe:
    language: str
    source: str
    entry: str

    @property
    def suffix(self) -> str:
        return SUFFIX[self.language]


def spell(kind: Type, language: str) -> str:
    if kind.opaque:
        if kind.opaque == "void":
            return "void"
        raise ProbeError(f"no probe for parameters of type {kind.opaque}")
    if language in ("crossgl", "glsl"):
        if not kind.shape:
            return kind.scalar
        prefix = {"float": "", "double": "d", "int": "i", "uint": "u", "bool": "b"}[
            kind.scalar
        ]
        if len(kind.shape) == 1:
            return f"{prefix}vec{kind.shape[0]}"
        if kind.scalar not in ("float", "double"):
            raise ProbeError(f"GLSL has no {kind.scalar} matrices")
        columns, rows = kind.shape
        return f"{prefix}mat{columns}" + ("" if columns == rows else f"x{rows}")
    if language == "metal" and kind.scalar == "double":
        raise ProbeError("Metal has no double type")
    if len(kind.shape) == 2:
        columns, rows = kind.shape
        # HLSL spells matrices rows x columns, Metal columns x rows.
        return (
            f"{kind.scalar}{rows}x{columns}"
            if language == "hlsl"
            else f"{kind.scalar}{columns}x{rows}"
        )
    return kind.scalar + (str(kind.shape[0]) if kind.shape else "")


def _storage(scalar: str) -> str:
    return "float" if scalar in ("bool", "double") else scalar


class _Builder:
    def __init__(self, language: str):
        self.language = language
        self.offsets: dict[str, int] = {}
        self.lines: list[str] = []

    def element(self, scalar: str) -> str:
        storage = _storage(scalar)
        offset = self.offsets.get(storage, 0)
        self.offsets[storage] = offset + 1
        read = f"{_BUFFERS[storage]}[i + {offset}u]"
        if scalar == "bool":
            return f"({read} > 0.0)"
        if scalar == "double":
            return f"(double){read}" if self.language == "hlsl" else f"double({read})"
        return read

    def value(self, kind: Type) -> str:
        scalar = Type(kind.scalar)
        if not kind.shape:
            return self.element(kind.scalar)
        if len(kind.shape) == 1:
            parts = [self.element(kind.scalar) for _ in range(kind.shape[0])]
            return f"{spell(kind, self.language)}({', '.join(parts)})"
        columns, rows = kind.shape
        # GLSL, CrossGL and Metal build matrices from columns, HLSL from rows.
        count, size = (rows, columns) if self.language == "hlsl" else (columns, rows)
        inner = Type(scalar.scalar, (size,))
        return f"{spell(kind, self.language)}({', '.join(self.value(inner) for _ in range(count))})"

    def body(self, name: str, signature: Signature) -> tuple[list[str], Type]:
        """Statements that call ``name`` once, and the element type of the output buffer."""
        args, output = [], None
        for index, param in enumerate(signature.params):
            local = f"a{index}"
            if param.direction == "inout":
                if output is not None:
                    raise ProbeError("more than one inout parameter")
                args.append("outp[i]")
                output = param.type
                continue
            if param.direction == "out":
                self.lines.append(f"{spell(param.type, self.language)} {local};")
            else:
                self.lines.append(
                    f"{spell(param.type, self.language)} {local} = {self.value(param.type)};"
                )
            args.append(local)
        call = f"{name}({', '.join(args)})"
        result = signature.result
        if output is not None or result.opaque:
            self.lines.append(f"{call};")
            if output is None:
                output = Type("uint")
                self.lines.append("outp[i] = 0u;")
            return self.lines, output
        self.lines.append(f"{spell(result, self.language)} r = {call};")
        if result.scalar == "bool":
            first = "r.x" if result.shape else "r"
            self.lines.append(f"outp[i] = {first} ? 1 : 0;")
            return self.lines, Type("int")
        stored = (
            result if len(result.shape) < 2 else Type(result.scalar, (result.shape[1],))
        )
        value = "r" if len(result.shape) < 2 else "r[0]"
        if result.scalar == "double":
            stored = stored.with_scalar("float")
            value = (
                f"({spell(stored, self.language)}){value}"
                if self.language == "hlsl"
                else f"{spell(stored, self.language)}({value})"
            )
        self.lines.append(f"outp[i] = {value};")
        return self.lines, stored


def _crossgl(buffers: list[tuple[str, str]], body: list[str]) -> str:
    lines = [
        "shader Probe {",
        "    compute {",
        "        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
    ]
    lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
    lines += ["        void main() {", "            uint i = gl_GlobalInvocationID.x;"]
    lines += [f"            {line}" for line in body]
    lines += ["        }", "    }", "}"]
    return "\n".join(lines) + "\n"


def _glsl(buffers: list[tuple[str, str]], body: list[str]) -> str:
    lines = [
        "#version 460",
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
    ]
    for binding, (kind, name) in enumerate(buffers):
        lines.append(
            f"layout(std430, binding = {binding}) buffer B{binding} {{ {kind} {name}[]; }};"
        )
    lines += ["void main() {", "    uint i = gl_GlobalInvocationID.x;"]
    lines += [f"    {line}" for line in body]
    return "\n".join(lines + ["}"]) + "\n"


def _hlsl(buffers: list[tuple[str, str]], body: list[str]) -> str:
    lines = [
        f"RWStructuredBuffer<{kind}> {name} : register(u{binding});"
        for binding, (kind, name) in enumerate(buffers)
    ]
    lines += [
        "[numthreads(1, 1, 1)]",
        "void main(uint3 id : SV_DispatchThreadID) {",
        "    uint i = id.x;",
    ]
    lines += [f"    {line}" for line in body]
    return "\n".join(lines + ["}"]) + "\n"


def _metal(buffers: list[tuple[str, str]], body: list[str]) -> str:
    params = [
        f"device {kind}* {name} [[buffer({binding})]]"
        for binding, (kind, name) in enumerate(buffers)
    ]
    params.append("uint3 id [[thread_position_in_grid]]")
    lines = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        f"kernel void probe({', '.join(params)}) {{",
        "    uint i = id.x;",
    ]
    lines += [f"    {line}" for line in body]
    return "\n".join(lines + ["}"]) + "\n"


_SKELETONS = {"crossgl": _crossgl, "glsl": _glsl, "hlsl": _hlsl, "metal": _metal}
ENTRIES = {"crossgl": "main", "glsl": "main", "hlsl": "main", "metal": "probe"}


def build(feature: Feature, language: str) -> Probe:
    """The probe for ``feature``: CrossGL for the output direction, the feature's own language for input."""
    if feature.signature is None:
        raise ProbeError(feature.note or "the feature list gives no signature")
    if feature.stage != "compute":
        raise ProbeError(f"no {feature.stage}-stage probe yet")
    name = feature.crossgl if language == "crossgl" else feature.name
    if not name:
        raise ProbeError(feature.note or "CrossGL has no spelling for this feature")
    builder = _Builder(language)
    body, output = builder.body(name, feature.signature)
    buffers = [
        (_storage(scalar), _BUFFERS[scalar])
        for scalar in ("float", "int", "uint")
        if scalar in builder.offsets
    ]
    buffers.append((spell(output, language), "outp"))
    return Probe(language, _SKELETONS[language](buffers, body), ENTRIES[language])

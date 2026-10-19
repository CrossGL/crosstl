"""Minimal probe programs: one call of one feature inside a compute kernel.

Arguments are read from input buffers, so no call folds to a constant, and the result is written to an output
buffer, so no call is dead code. Output probes are CrossGL (:mod:`.crossgl`); input probes are written in the source
language. A *baseline* probe reads the same arguments but makes no call; :mod:`.semantic` subtracts it to find the
instructions the call itself lowers to.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import constructs
from .crossgl import DIALECT as CROSSGL
from .dialect import Buffer, Dialect, NoSpelling, ProbeError
from .model import Feature, Signature, Type

__all__ = ["DIALECTS", "NoSpelling", "Probe", "ProbeError", "build", "spell"]
_BUFFERS = {"float": "fin", "int": "iin", "uint": "uin"}


class GLSL(Dialect):
    language = "glsl"
    suffix = ".comp"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            "#version 460",
            *parts.get("prelude", ()),
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
        ]
        for binding, (kind, name) in enumerate(buffers):
            lines.append(
                f"layout(std430, binding = {binding}) buffer B{binding} {{ {kind} {name}[]; }};"
            )
        lines += parts.get("declarations", ())
        lines += ["void main() {", "    uint i = gl_GlobalInvocationID.x;"]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"


class HLSL(Dialect):
    language = "hlsl"
    suffix = ".hlsl"
    rows_first = True

    def vector(self, scalar: str, size: int) -> str:
        return f"{scalar}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        return f"{scalar}{rows}x{columns}"

    def cast(self, kind: Type, value: str) -> str:
        return f"({self.type(kind)}){value}"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = list(parts.get("prelude", ()))
        lines += [
            f"RWStructuredBuffer<{kind}> {name} : register(u{binding});"
            for binding, (kind, name) in enumerate(buffers)
        ]
        lines += parts.get("declarations", ())
        params = ", ".join(["uint3 id : SV_DispatchThreadID", *parts.get("params", ())])
        lines += [
            "[numthreads(1, 1, 1)]",
            f"void main({params}) {{",
            "    uint i = id.x;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"


class Metal(Dialect):
    language = "metal"
    suffix = ".metal"
    entry = "probe"

    def scalar(self, scalar: str) -> str:
        if scalar == "double":
            raise ProbeError("Metal has no double type")
        return scalar

    def vector(self, scalar: str, size: int) -> str:
        return f"{self.scalar(scalar)}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        if scalar != "float":
            raise ProbeError(f"Metal has no {scalar} matrices")
        return f"float{columns}x{rows}"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = [
            f"device {kind}* {name} [[buffer({binding})]]"
            for binding, (kind, name) in enumerate(buffers)
        ]
        params.append("uint3 id [[thread_position_in_grid]]")
        params += parts.get("params", ())
        lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            *parts.get("prelude", ()),
            "",
            *parts.get("declarations", ()),
            f"kernel void probe({', '.join(params)}) {{",
            "    uint i = id.x;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"


class WGSL(Dialect):
    language = "wgsl"
    suffix = ".wgsl"
    _SCALARS = {"float": "f32", "int": "i32", "uint": "u32", "bool": "bool"}

    def scalar(self, scalar: str) -> str:
        if scalar not in self._SCALARS:
            raise ProbeError(f"WGSL has no {scalar} type")
        return self._SCALARS[scalar]

    def vector(self, scalar: str, size: int) -> str:
        return f"vec{size}<{self.scalar(scalar)}>"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        if scalar != "float":
            raise ProbeError(f"WGSL has no {scalar} matrices")
        return f"mat{columns}x{rows}<f32>"

    def declare(self, kind: Type, name: str, value: str = "") -> str:
        initial = f" = {value}" if value else ""
        return f"var {name}: {self.type(kind)}{initial};"

    def out_argument(self, name: str) -> str:
        raise ProbeError("WGSL has no output parameters")

    def inout_argument(self) -> str:
        return "&outp[i]"

    def to_int(self, condition: str) -> str:
        return f"select(0, 1, {condition})"

    def buffer_element(self, kind: Type, atomic: bool) -> str:
        return f"atomic<{self.type(kind)}>" if atomic else self.type(kind)

    def keep(self, atomic: bool) -> str:
        # An atomic buffer element can only be used through atomic built-in functions.
        return "" if atomic else self.store("outp[i]")

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = list(parts.get("prelude", ()))
        lines += [
            f"@group(0) @binding({binding}) var<storage, read_write> {name}: array<{kind}>;"
            for binding, (kind, name) in enumerate(buffers)
        ]
        lines += parts.get("declarations", ())
        params = ", ".join(
            ["@builtin(global_invocation_id) id: vec3<u32>", *parts.get("params", ())]
        )
        lines += [
            "@compute @workgroup_size(1, 1, 1)",
            f"fn main({params}) {{",
            "    let i = id.x;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"


DIALECTS: dict[str, Dialect] = {
    d.language: d for d in (CROSSGL, GLSL(), HLSL(), Metal(), WGSL())
}


@dataclass(frozen=True)
class Probe:
    language: str
    source: str
    entry: str
    suffix: str


def spell(kind: Type, language: str) -> str:
    return DIALECTS[language].type(kind)


def _storage(scalar: str) -> str:
    return "float" if scalar in ("bool", "double") else scalar


class _Builder:
    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.offsets: dict[str, int] = {}
        self.lines: list[str] = []

    def element(self, scalar: str) -> str:
        storage = _storage(scalar)
        offset = self.offsets.get(storage, 0)
        self.offsets[storage] = offset + 1
        read = self.dialect.read(_BUFFERS[storage], offset)
        if scalar == "bool":
            return self.dialect.positive(read)
        if scalar == "double":
            return self.dialect.cast(Type("double"), read)
        return read

    def value(self, kind: Type) -> str:
        if not kind.shape:
            return self.element(kind.scalar)
        if len(kind.shape) == 1:
            parts = [self.element(kind.scalar) for _ in range(kind.shape[0])]
            return self.dialect.construct(kind, parts)
        columns, rows = kind.shape
        count, size = (rows, columns) if self.dialect.rows_first else (columns, rows)
        inner = Type(kind.scalar, (size,))
        return self.dialect.construct(kind, [self.value(inner) for _ in range(count)])

    def _arguments(self, signature: Signature) -> tuple[list[str], Type | None]:
        dialect, arguments, output = self.dialect, [], None
        for kind in (signature.result, *(p.type for p in signature.params)):
            dialect.type(kind)  # raises ProbeError for what the language cannot spell
        for index, param in enumerate(signature.params):
            local = f"a{index}"
            if param.direction == "inout":
                if output is not None:
                    raise ProbeError("more than one inout parameter")
                arguments.append(dialect.inout_argument())
                output = param.type
            elif param.direction == "out":
                self.lines.append(dialect.declare(param.type, local))
                arguments.append(dialect.out_argument(local))
            elif param.direction == "const":
                arguments.append(dialect.literal(param.type))
            else:
                value = self.value(param.type)
                self.lines.append(dialect.declare(param.type, local, value))
                arguments.append(local)
        return arguments, output

    def body(
        self, name: str, signature: Signature, baseline: bool = False
    ) -> tuple[list[str], Type]:
        """Statements that call ``name`` once, and the element type of the output buffer."""
        dialect = self.dialect
        arguments, output = self._arguments(signature)
        atomic = output is not None
        call = dialect.call(name, arguments)
        result = signature.result
        if output is not None or result.opaque:
            stores = [dialect.statement(call)]
            if output is None:
                output = Type("uint")
                stores.append(dialect.store("0u"))
        elif result.scalar == "bool":
            flag = dialect.first("r") if result.shape else "r"
            stores = [
                dialect.declare(result, "r", call),
                dialect.store(dialect.to_int(flag)),
            ]
            output = Type("int")
        else:
            output, value = result, "r"
            if len(result.shape) == 2:
                columns, rows = result.shape
                output = Type(result.scalar, (columns if dialect.rows_first else rows,))
                value = dialect.part("r")
            if result.scalar == "double":
                output = output.with_scalar("float")
                value = dialect.cast(output, value)
            stores = [dialect.declare(result, "r", call), dialect.store(value)]
        if baseline:
            stores = [line for line in [dialect.keep(atomic)] if line]
        return self.lines + stores, output


def build(feature: Feature, language: str, *, baseline: bool = False) -> Probe:
    """The probe for ``feature`` in ``language``: CrossGL for the output direction, the feature's own language for
    input and for lowering to SPIR-V."""
    if constructs.is_construct(feature):
        if baseline:
            raise ProbeError("language-level features have no baseline probe")
        dialect = DIALECTS[language]
        source, entry, suffix = constructs.build(feature, language, dialect)
        return Probe(language, source, entry, suffix)
    if feature.signature is None:
        raise ProbeError(feature.note or "the feature list gives no signature")
    if feature.stage != "compute":
        raise ProbeError(f"no {feature.stage}-stage probe yet")
    name = (
        feature.crossgl if language == "crossgl" else feature.spelling or feature.name
    )
    if not name:
        raise NoSpelling(
            "CrossGL has no spelling for this operation"
            + (f": {feature.note}" if feature.note else "")
        )
    dialect = DIALECTS[language]
    builder = _Builder(dialect)
    body, output = builder.body(name, feature.signature, baseline)
    buffers = [
        (dialect.type(Type(scalar)), _BUFFERS[scalar])
        for scalar in ("float", "int", "uint")
        if scalar in builder.offsets
    ]
    atomic = any(p.direction == "inout" for p in feature.signature.params)
    buffers.append((dialect.buffer_element(output, atomic), "outp"))
    return Probe(language, dialect.kernel(buffers, body), dialect.entry, dialect.suffix)

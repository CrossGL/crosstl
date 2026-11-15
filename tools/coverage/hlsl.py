"""The HLSL dialect, and Slang's, which compiles HLSL kernels with its own entry-point attributes and types."""

from __future__ import annotations

from .dialect import Buffer, Dialect
from .model import (
    COOP_COLUMNS,
    DESCRIPTOR_HANDLE,
    RAYTRACING_STAGES,
    SLANG_ENUMERATORS,
    Type,
    coop_element,
    descriptor_reading,
    slang_literal,
    tuple_element,
)
from .objects import HIT_OBJECT, OBJECTS, RECORD, STREAM, VERTEX


class HLSL(Dialect):
    language = "hlsl"
    suffix = ".hlsl"
    rows_first = True
    # ``{lanes}`` is the workgroup size (:meth:`.Dialect.kernel`).
    entry_attributes: tuple[str, ...] = ("[numthreads({lanes}, 1, 1)]",)
    fragment_attributes: tuple[str, ...] = ()
    # The validators compile with 16-bit types enabled, so ``half`` is a 16-bit float (HLSL shader model 6.2).
    wide_integers = True
    narrow_scalars = True
    packed_bytes = True
    _SCALARS = {
        "int64": "int64_t",
        "uint64": "uint64_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int8x4_packed": "int8_t4_packed",
        "uint8x4_packed": "uint8_t4_packed",
    }

    def scalar(self, scalar: str) -> str:
        return self._SCALARS.get(scalar, scalar)

    def vector(self, scalar: str, size: int) -> str:
        return f"{self.scalar(scalar)}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        return f"{self.scalar(scalar)}{rows}x{columns}"

    def cast(self, kind: Type, value: str) -> str:
        return f"({self.type(kind)}){value}"

    def buffer(self, kind: str, name: str, binding: int) -> str:
        return f"RWStructuredBuffer<{kind}> {name} : register(u{binding});"

    def resource(self, spelling: str, name: str, register: str) -> str:
        return f"{spelling} {name} : register({register});"

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        return self.resource(spelling, name, f"t{binding}")

    def workgroup(self, kind: Type, name: str) -> str:
        count = f"[{kind.count}]" if kind.count else ""
        return f"groupshared {self.type(kind.element)} {name}{count};"

    def flat(self, kind: str) -> str:
        return f"nointerpolation {kind}"

    def _globals(self, buffers: list[Buffer], parts: dict) -> list[str]:
        lines = list(parts.get("prelude", ()))
        lines += [
            self.buffer(kind, name, binding)
            for binding, (kind, name) in enumerate(buffers)
        ]
        return lines + list(parts.get("declarations", ()))

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = ", ".join(["uint3 id : SV_DispatchThreadID", *parts.get("params", ())])
        lines = [
            *self._globals(buffers, parts),
            *(a.format(lanes=parts.get("lanes", 1)) for a in self.entry_attributes),
            f"void main({params}) {{",
            "    uint i = id.x;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        inputs = ", ".join(
            f"{kind} {name} : TEXCOORD{index}"
            for index, (kind, name) in enumerate(parts.get("inputs", ()))
        )
        lines = [
            *self._globals(buffers, parts),
            *self.fragment_attributes,
            f"float4 main({inputs}) : SV_Target {{",
            "    uint i = 0u;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["    return float4(0.0, 0.0, 0.0, 0.0);", "}"]) + "\n"

    def geometry(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        # A Direct3D geometry shader always takes an output stream. Its vertex has no system value, which the
        # validator would require every path to write, so a probe that appends nothing is valid.
        stream = parts.get("parameter", STREAM)
        lines = [
            f"struct {VERTEX} {{",
            "    float4 value : TEXCOORD0;",
            "};",
            *self._globals(buffers, parts),
            *self.stage_attributes.get("geometry", ()),
            "[maxvertexcount(1)]",
            f"void main(point float4 position[1] : POSITION, inout {stream} obj) {{",
            "    uint i = 0u;",
            f"    {VERTEX} {self.vertex};",
            f"    {self.vertex}.value = position[0];",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    # Hit shaders take the ray payload, which shader model 6.7 and later require to declare who reads and writes each
    # field, and the triangle's barycentrics.
    _PAYLOAD = (
        "struct [raypayload] Payload",
        "{",
        "    float4 color : read(caller) : write(anyhit, closesthit);",
        "};",
    )
    _HIT_PARAMETERS = (
        "inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes"
    )

    def raytracing(
        self, stage: str, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        hit = stage in ("any-hit", "closest-hit")
        # A ray-generation probe that traces declares its payload as a local.
        traced = parts.get("traced", False)
        lines = [
            *(self._PAYLOAD if hit or traced else ()),
            *self._globals(buffers, parts),
            f'[shader("{RAYTRACING_STAGES[stage]}")]',
            f"void main({self._HIT_PARAMETERS if hit else ''}) {{",
            "    uint i = 0u;",
            *([f"    Payload {self.payload};"] if traced else ()),
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    # Entry-point attributes that name a graphics stage, which Slang needs and HLSL takes from the profile instead.
    stage_attributes: dict[str, tuple[str, ...]] = {}
    _PATCH = (
        "struct ControlPoint {",
        "    float4 position : POSITION;",
        "};",
        "struct PatchConstants {",
        "    float edges[3] : SV_TessFactor;",
        "    float inside : SV_InsideTessFactor;",
        "};",
    )

    def tessellation_control(
        self, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        lines = [
            *self._PATCH,
            *self._globals(buffers, parts),
            "PatchConstants constants(InputPatch<ControlPoint, 3> patch) {",
            "    PatchConstants output;",
            "    output.edges[0] = 1.0;",
            "    output.edges[1] = 1.0;",
            "    output.edges[2] = 1.0;",
            "    output.inside = 1.0;",
            "    uint i = 0u;",
            *(f"    {line}" for line in body),
            "    return output;",
            "}",
            *self.stage_attributes.get("tessellation-control", ()),
            '[domain("tri")]',
            '[partitioning("integer")]',
            '[outputtopology("triangle_cw")]',
            "[outputcontrolpoints(3)]",
            '[patchconstantfunc("constants")]',
            "ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id : SV_OutputControlPointID) {",
            "    ControlPoint output;",
            "    output.position = patch[id].position * 2.0;",
            "    return output;",
            "}",
        ]
        return "\n".join(lines) + "\n"

    def mesh(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            "struct Vertex {",
            "    float4 position : SV_Position;",
            "};",
            *self._globals(buffers, parts),
            *self.stage_attributes.get("mesh", ()),
            "[numthreads(1, 1, 1)]",
            '[outputtopology("triangle")]',
            "void main(out vertices Vertex verts[3], out indices uint3 tris[1]) {",
            "    uint i = 0u;",
            *(f"    {line}" for line in body),
            "    verts[0].position = float4(0.0, 0.0, 0.0, 1.0);",
            "    verts[1].position = float4(1.0, 0.0, 0.0, 1.0);",
            "    verts[2].position = float4(0.0, 1.0, 0.0, 1.0);",
            "    tris[0] = uint3(0, 1, 2);",
            "}",
        ]
        return "\n".join(lines) + "\n"

    def task(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            "struct Payload {",
            "    uint value;",
            "};",
            f"groupshared Payload {self.payload};",
            *self._globals(buffers, parts),
            *self.stage_attributes.get("task", ()),
            "[numthreads(1, 1, 1)]",
            "void main() {",
            "    uint i = 0u;",
            *(f"    {line}" for line in body),
            "}",
        ]
        return "\n".join(lines) + "\n"

    # Node inputs that make a node a coalescing one, which runs a thread group per batch of input records; any other
    # node is a broadcasting one with a fixed dispatch grid. A records object is requested from a node output.
    _COALESCING = ("EmptyNodeInput", "GroupNodeInputRecords")
    _RECORDS = {
        "ThreadNodeOutputRecords": "GetThreadNodeOutputRecords",
        "GroupNodeOutputRecords": "GetGroupNodeOutputRecords",
    }

    def holding(self, opaque: str) -> str:
        if opaque == HIT_OBJECT:
            return OBJECTS[HIT_OBJECT].spelling
        return f"{opaque}<{RECORD}>" if opaque in self._RECORDS else ""

    def node(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        parameter = parts.get("parameter", "")
        kind = parameter.split("<")[0]
        # Only a record whose type tracks it can signal FinishedCrossGroupSharing.
        shared = kind == "RWDispatchNodeInputRecord"
        declared, opening = f"[MaxRecords(1)] {parameter} obj", []
        if kind in self._RECORDS:
            declared = f"[MaxRecords(1)] NodeOutput<{RECORD}> output"
            opening = [f"    {parameter} obj = output.{self._RECORDS[kind]}(1);"]
        elif shared:
            declared = f"globallycoherent {parameter} obj"
        coalescing = kind in self._COALESCING
        lines = [
            f"struct {'[NodeTrackRWInputSharing] ' if shared else ''}{RECORD} {{",
            "    uint value;",
            "};",
            *self._globals(buffers, parts),
            '[Shader("node")]',
            f'[NodeLaunch("{"coalescing" if coalescing else "broadcasting"}")]',
            *(() if coalescing else ("[NodeDispatchGrid(1, 1, 1)]",)),
            "[NumThreads(1, 1, 1)]",
            f"void main({declared if parameter else ''}) {{",
            "    uint i = 0u;",
            *opening,
            *(f"    {line}" for line in body),
            "}",
        ]
        return "\n".join(lines) + "\n"


class Slang(HLSL):
    """Slang compiles HLSL kernels; the ``shader`` attribute marks the entry point, so slangc needs no ``-entry``."""

    language = "slang"
    suffix = ".slang"
    string = "String"
    entry_attributes = ('[shader("compute")]', "[numthreads({lanes}, 1, 1)]")
    fragment_attributes = ('[shader("fragment")]',)
    stage_attributes = {
        "geometry": ('[shader("geometry")]',),
        "tessellation-control": ('[shader("hull")]',),
        "mesh": ('[shader("mesh")]',),
        "task": ('[shader("amplification")]',),
    }
    # Slang declares a ray payload as a plain struct; payload access qualifiers are an HLSL shader model 6.7 rule.
    _PAYLOAD = ("struct Payload", "{", "    float4 color;", "};")

    def buffer(self, kind: str, name: str, binding: int) -> str:
        # slangc assigns bindings in declaration order; a D3D register would only draw a warning for SPIR-V.
        return f"RWStructuredBuffer<{kind}> {name};"

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        # As for buffers; slangc also rejects a t register for a writable resource such as RWByteAddressBuffer. A
        # descriptor handle is ordinary data, which slangc warns about unless it is declared uniform.
        uniform = "uniform " if spelling == DESCRIPTOR_HANDLE else ""
        return f"{uniform}{spelling} {name};"

    def type(self, kind: Type) -> str:
        if kind == COOP_COLUMNS:
            return super().type(kind.with_scalar("int"))
        if coop_element(kind.opaque):
            return f"CoopVec<{self.scalar(kind.scalar)}, {kind.shape[0]}>"
        if kind.opaque in SLANG_ENUMERATORS:
            return kind.opaque
        held = tuple_element(kind.opaque)
        if held:
            return f"Tuple<{', '.join([self.scalar(kind.scalar)] * held[1])}>"
        return super().type(kind)

    def literal(self, kind: Type) -> str:
        return slang_literal(kind) or super().literal(kind)

    def pointer(self, kind: Type, name: str) -> str:
        # A global pointer is a uniform that holds a device address; slangc lowers it to a SPIR-V physical storage
        # buffer pointer.
        return f"uniform {self.type(kind)}* {name};"

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        # A cooperative vector result is read through its first element, and a tuple through its first member.
        element = coop_element(opaque)
        if element:
            return element, "{}[0]"
        held = tuple_element(opaque)
        if held:
            return held[0], "{}._0"
        return descriptor_reading(opaque) or super().reading(opaque)

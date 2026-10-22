"""The dialect of each language a probe can be written in. The shading languages are here, the compute-kernel
languages in :mod:`.kernels` and CrossGL's own dialect in :mod:`.crossgl`."""

from __future__ import annotations

from .crossgl import DIALECT as CROSSGL
from .crossgl import ES_DIALECT as CROSSGL_ES
from .dialect import UNIFORM_ARRAY_LENGTH, Buffer, Dialect, ProbeError, uniform_read
from .kernels import CUDA, HIP, Mojo, OpenCL, Rust
from .model import (
    COOP_COLUMNS,
    COOP_ENUMERATORS,
    RAYTRACING_STAGES,
    Type,
    coop_element,
    coop_literal,
    tuple_element,
)
from .objects import HIT_OBJECT, OBJECTS, RECORD, STREAM, VERTEX


class GLSL(Dialect):
    language = "glsl"
    suffix = ".comp"
    fragment_suffix = ".frag"
    geometry_suffix = ".geom"

    def _globals(self, buffers: list[Buffer], parts: dict) -> list[str]:
        lines = [
            f"layout(std430, binding = {binding}) buffer B{binding} {{ {kind} {name}[]; }};"
            for binding, (kind, name) in enumerate(buffers)
        ]
        return lines + list(parts.get("declarations", ()))

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        layout = (
            f"{qualifier}, binding = {binding}" if qualifier else f"binding = {binding}"
        )
        return f"layout({layout}) uniform {spelling} {name};"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            "#version 460",
            *parts.get("prelude", ()),
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
            *self._globals(buffers, parts),
        ]
        lines += ["void main() {", "    uint i = gl_GlobalInvocationID.x;"]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            "#version 460",
            *parts.get("prelude", ()),
            *self._globals(buffers, parts),
            *(
                f"layout(location = {location}) in {kind} {name};"
                for location, (kind, name) in enumerate(parts.get("inputs", ()))
            ),
            "layout(location = 0) out vec4 color;",
            "void main() {",
            "    uint i = 0u;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["    color = vec4(0.0);", "}"]) + "\n"

    def geometry(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            "#version 460",
            *parts.get("prelude", ()),
            "layout(points) in;",
            "layout(points, max_vertices = 1) out;",
            *self._globals(buffers, parts),
            "void main() {",
            "    uint i = 0u;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"


class HLSL(Dialect):
    language = "hlsl"
    suffix = ".hlsl"
    rows_first = True
    entry_attributes: tuple[str, ...] = ("[numthreads(1, 1, 1)]",)
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
            *self.entry_attributes,
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


class Metal(Dialect):
    language = "metal"
    suffix = ".metal"
    entry = "probe"
    wide_integers = True
    narrow_scalars = True
    _SCALARS = {
        "int64": "long",
        "uint64": "ulong",
        "int16": "short",
        "uint16": "ushort",
    }
    # The constant a probe passes for each enumeration a Metal function takes (features.metal.ENUMS). Metal allows
    # only relaxed ordering on device atomics, and mem_none makes a barrier an execution barrier only.
    _ENUMERATORS = {
        "memory_order": "memory_order_relaxed",
        "mem_flags": "mem_flags::mem_none",
    }
    # A ballot's mask converts to its vote_t integer only explicitly: a 64-bit simd_vote::vote_t and a 16-bit
    # quad_vote::vote_t, which the probe stores as ulong and uint.
    _VOTES = {
        "simd_vote": (Type("uint64"), "static_cast<simd_vote::vote_t>({})"),
        "quad_vote": (Type("uint"), "static_cast<quad_vote::vote_t>({})"),
    }

    def type(self, kind: Type) -> str:
        if kind.opaque in self._ENUMERATORS:
            return kind.opaque
        return super().type(kind)

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        return self._VOTES.get(opaque)

    def scalar(self, scalar: str) -> str:
        if scalar == "double":
            raise ProbeError("Metal has no double type")
        return self._SCALARS.get(scalar, scalar)

    def literal(self, kind: Type) -> str:
        if kind.opaque in self._ENUMERATORS:
            return self._ENUMERATORS[kind.opaque]
        return super().literal(kind)

    def buffer_element(self, kind: Type, atomic: bool) -> str:
        return f"atomic_{self.type(kind)}" if atomic else self.type(kind)

    def inout_argument(self) -> str:
        return "&outp[i]"

    def address(self, name: str) -> str:
        return f"&{name}"

    def keep(self, atomic: bool) -> str:
        # An atomic buffer element can only be used through atomic functions.
        return "" if atomic else self.store("outp[i]")

    def vector(self, scalar: str, size: int) -> str:
        return f"{self.scalar(scalar)}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        if scalar not in ("float", "half"):
            raise ProbeError(f"Metal has no {scalar} matrices")
        return f"{scalar}{columns}x{rows}"

    def _params(self, buffers: list[Buffer]) -> list[str]:
        return [
            f"device {kind}* {name} [[buffer({binding})]]"
            for binding, (kind, name) in enumerate(buffers)
        ]

    def _head(self, parts: dict) -> list[str]:
        return [
            "#include <metal_stdlib>",
            "using namespace metal;",
            *parts.get("prelude", ()),
            "",
            *parts.get("declarations", ()),
        ]

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = self._params(buffers)
        params.append("uint3 id [[thread_position_in_grid]]")
        params += parts.get("params", ())
        lines = [
            *self._head(parts),
            f"kernel void probe({', '.join(params)}) {{",
            "    uint i = id.x;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = [*self._params(buffers), *parts.get("params", ())]
        lines = [
            *self._head(parts),
            f"fragment float4 probe({', '.join(params)}) {{",
            "    uint i = 0u;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["    return float4(0.0);", "}"]) + "\n"


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

    def workgroup(self, kind: Type, name: str) -> str:
        return f"var<workgroup> {name}: {self.type(kind)};"

    def workgroup_argument(self, name: str) -> str:
        return f"&{name}"

    def inout_argument(self) -> str:
        return "&outp[i]"

    def buffer_argument(self, name: str) -> str:
        return f"&{name}"

    def to_int(self, condition: str) -> str:
        return f"select(0, 1, {condition})"

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        return f"@group(0) @binding({binding}) var {name}: {spelling};"

    def buffer_element(self, kind: Type, atomic: bool) -> str:
        return f"atomic<{self.type(kind)}>" if atomic else self.type(kind)

    def keep(self, atomic: bool) -> str:
        # An atomic buffer element can only be used through atomic built-in functions.
        return "" if atomic else self.store("outp[i]")

    def _globals(self, buffers: list[Buffer], parts: dict) -> list[str]:
        lines = list(parts.get("prelude", ()))
        lines += [
            f"@group(0) @binding({binding}) var<storage, read_write> {name}: array<{kind}>;"
            for binding, (kind, name) in enumerate(buffers)
        ]
        return lines + list(parts.get("declarations", ()))

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = ", ".join(
            ["@builtin(global_invocation_id) id: vec3<u32>", *parts.get("params", ())]
        )
        lines = [
            *self._globals(buffers, parts),
            "@compute @workgroup_size(1, 1, 1)",
            f"fn main({params}) {{",
            "    let i = id.x;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            *self._globals(buffers, parts),
            "@fragment",
            f"fn main({', '.join(parts.get('params', ()))}) -> @location(0) vec4<f32> {{",
            "    let i = 0u;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["    return vec4<f32>(0.0);", "}"]) + "\n"


class WebGL(Dialect):
    """GLSL ES 3.00, the shading language of WebGL 2. It has no compute stage and no storage buffers, so every probe is
    a fragment shader that reads uniform arrays and writes a local array (:data:`.dialect.UNIFORM_ARRAY_LENGTH`).
    """

    language = "webgl"
    suffix = ".frag"
    _HEADER = ("#version 300 es", "precision highp float;", "precision highp int;")
    _SCALARS = frozenset({"float", "int", "uint", "bool"})

    def type(self, kind: Type) -> str:
        if not kind.opaque and kind.scalar not in self._SCALARS:
            raise ProbeError(f"GLSL ES 3.00 has no {kind.scalar} type")
        return super().type(kind)

    def read(self, buffer: str, offset: int) -> str:
        return uniform_read(buffer, offset)

    # GLSL ES 3.00 has no binding layouts, and no default precision for most sampler types.
    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        if qualifier:
            raise ProbeError("GLSL ES 3.00 has no images or subpass inputs")
        return f"uniform highp {spelling} {name};"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        *inputs, (output, _) = buffers
        lines = [
            *self._HEADER,
            *parts.get("prelude", ()),
            *(
                f"uniform {kind} {name}[{UNIFORM_ARRAY_LENGTH}];"
                for kind, name in inputs
            ),
            *parts.get("declarations", ()),
            "layout(location = 0) out vec4 color;",
            "void main() {",
            "    color = vec4(0.0);",
            f"    {output} outp[1];",
            "    uint i = 0u;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        if parts.get("inputs"):
            raise ProbeError("no shader-input probe for GLSL ES yet")
        return self.kernel(buffers, body, **parts)


class Slang(HLSL):
    """Slang compiles HLSL kernels; the ``shader`` attribute marks the entry point, so slangc needs no ``-entry``."""

    language = "slang"
    suffix = ".slang"
    string = "String"
    entry_attributes = ('[shader("compute")]', "[numthreads(1, 1, 1)]")
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
        # As for buffers; slangc also rejects a t register for a writable resource such as RWByteAddressBuffer.
        return f"{spelling} {name};"

    def type(self, kind: Type) -> str:
        if kind == COOP_COLUMNS:
            return super().type(kind.with_scalar("int"))
        if coop_element(kind.opaque):
            return f"CoopVec<{self.scalar(kind.scalar)}, {kind.shape[0]}>"
        if kind.opaque in COOP_ENUMERATORS:
            return kind.opaque
        held = tuple_element(kind.opaque)
        if held:
            return f"Tuple<{', '.join([self.scalar(kind.scalar)] * held[1])}>"
        return super().type(kind)

    def literal(self, kind: Type) -> str:
        return coop_literal(kind) or super().literal(kind)

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
        return (held[0], "{}._0") if held else super().reading(opaque)


DIALECTS: dict[str, Dialect] = {
    d.language: d
    for d in (
        CROSSGL,
        CUDA(),
        GLSL(),
        HIP(),
        HLSL(),
        Metal(),
        Mojo(),
        OpenCL(),
        Rust(),
        Slang(),
        WebGL(),
        WGSL(),
    )
}
# WebGL's output probes are written in the CrossGL dialect that GLSL ES can run: see :class:`.crossgl.CrossGLES`.
DIALECTS["crossgl-es"] = CROSSGL_ES

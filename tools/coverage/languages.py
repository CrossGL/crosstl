"""The dialect of each language a probe can be written in. The shading languages are here, the compute-kernel
languages in :mod:`.kernels` and CrossGL's own dialect in :mod:`.crossgl`."""

from __future__ import annotations

from .crossgl import DIALECT as CROSSGL
from .dialect import Buffer, Dialect, ProbeError
from .kernels import CUDA, HIP, Mojo, Rust
from .model import RAYTRACING_STAGES, Type


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

    def uniform(self, spelling: str, name: str, binding: int, image_format: str) -> str:
        layout = (
            f"{image_format}, binding = {binding}"
            if image_format
            else f"binding = {binding}"
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
        lines = [
            *(self._PAYLOAD if hit else ()),
            *self._globals(buffers, parts),
            f'[shader("{RAYTRACING_STAGES[stage]}")]',
            f"void main({self._HIT_PARAMETERS if hit else ''}) {{",
            "    uint i = 0u;",
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

    def type(self, kind: Type) -> str:
        if kind.opaque in self._ENUMERATORS:
            return kind.opaque
        return super().type(kind)

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

    def inout_argument(self) -> str:
        return "&outp[i]"

    def to_int(self, condition: str) -> str:
        return f"select(0, 1, {condition})"

    def uniform(self, spelling: str, name: str, binding: int, image_format: str) -> str:
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


class Slang(HLSL):
    """Slang compiles HLSL kernels; the ``shader`` attribute marks the entry point, so slangc needs no ``-entry``."""

    language = "slang"
    suffix = ".slang"
    entry_attributes = ('[shader("compute")]', "[numthreads(1, 1, 1)]")
    fragment_attributes = ('[shader("fragment")]',)
    stage_attributes = {
        "tessellation-control": ('[shader("hull")]',),
        "mesh": ('[shader("mesh")]',),
        "task": ('[shader("amplification")]',),
    }
    # Slang declares a ray payload as a plain struct; payload access qualifiers are an HLSL shader model 6.7 rule.
    _PAYLOAD = ("struct Payload", "{", "    float4 color;", "};")

    def buffer(self, kind: str, name: str, binding: int) -> str:
        # slangc assigns bindings in declaration order; a D3D register would only draw a warning for SPIR-V.
        return f"RWStructuredBuffer<{kind}> {name};"


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
        Rust(),
        Slang(),
        WGSL(),
    )
}

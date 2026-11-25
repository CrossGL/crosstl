"""The dialect of each language a probe can be written in. GLSL, GLSL ES, Metal, WGSL and WebGL are here. HLSL and
Slang are in :mod:`.hlsl`, CUDA, HIP and Mojo in :mod:`.kernels`, Rust in :mod:`.rust`, OpenCL C in :mod:`.opencl` and
CrossGL in :mod:`.crossgl`."""

from __future__ import annotations

from .crossgl import DIALECT as CROSSGL
from .crossgl import ES_DIALECT as CROSSGL_ES
from .dialect import UNIFORM_ARRAY_LENGTH, Buffer, Dialect, ProbeError, uniform_read
from .hlsl import HLSL, Slang
from .kernels import CUDA, HIP, Mojo
from .model import Type
from .opencl import OpenCL
from .rust import Rust
from .templates.gles import PRECISION as ES_PRECISION
from .templates.gles import VERSION as ES_VERSION
from .templates.gles import with_precisions


class GLSL(Dialect):
    language = "glsl"
    suffix = ".comp"
    fragment_suffix = ".frag"
    geometry_suffix = ".geom"
    version = "#version 460"
    # Default precision statements, which follow the extensions a probe enables.
    precision: tuple[str, ...] = ()

    def _head(self, parts: dict) -> list[str]:
        return [self.version, *parts.get("prelude", ()), *self.precision]

    def _globals(self, buffers: list[Buffer], parts: dict) -> list[str]:
        read_only = parts.get("read_only", ())
        lines = [
            f"layout(std430, binding = {binding}) "
            + ("readonly " if name in read_only else "")
            + f"buffer B{binding} {{ {kind} {name}[]; }};"
            for binding, (kind, name) in enumerate(buffers)
        ]
        return lines + list(parts.get("declarations", ()))

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        layout = (
            f"{qualifier}, binding = {binding}" if qualifier else f"binding = {binding}"
        )
        return f"layout({layout}) uniform {spelling} {name};"

    def enable(self, extension: str) -> list[str]:
        # A feature may need several extensions, separated by spaces: subgroupQuadAll needs
        # GL_KHR_shader_subgroup_vote and GL_EXT_shader_quad_control.
        return [f"#extension {name} : require" for name in extension.split()]

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lanes = parts.get("lanes", 1)
        lines = [
            *self._head(parts),
            f"layout(local_size_x = {lanes}, local_size_y = 1, local_size_z = 1) in;",
            *self._globals(buffers, parts),
        ]
        lines += ["void main() {", "    uint i = gl_GlobalInvocationID.x;"]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = [
            *self._head(parts),
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
            *self._head(parts),
            "layout(points) in;",
            "layout(points, max_vertices = 1) out;",
            *self._globals(buffers, parts),
            "void main() {",
            "    uint i = 0u;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"


class GLES(GLSL):
    """GLSL ES 3.20, the shading language of OpenGL ES 3.2. A probe is GLSL's under its version, with the default
    precision of floats and integers declared, since a fragment shader has none for floats, and with a precision on
    every opaque uniform, since most opaque types have no default (GLSL ES 3.20, section 4.7.4).
    """

    language = "gles"
    version = ES_VERSION
    precision = ES_PRECISION

    def scalar(self, scalar: str) -> str:
        if scalar == "double":
            raise ProbeError("GLSL ES 3.20 has no double type")
        return super().scalar(scalar)

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        return with_precisions(super().kernel(buffers, body, **parts))

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        return with_precisions(super().fragment(buffers, body, **parts))

    def geometry(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        return with_precisions(super().geometry(buffers, body, **parts))

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        layout = (
            f"{qualifier}, binding = {binding}" if qualifier else f"binding = {binding}"
        )
        return f"layout({layout}) uniform highp {spelling} {name};"


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

    def _params(self, buffers: list[Buffer], parts: dict) -> list[str]:
        read_only = parts.get("read_only", ())
        return [
            ("const " if name in read_only else "")
            + f"device {kind}* {name} [[buffer({binding})]]"
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
        params = self._params(buffers, parts)
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
        params = [*self._params(buffers, parts), *parts.get("params", ())]
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

    def enable(self, extension: str) -> list[str]:
        lines = [f"enable {extension};"]
        if extension == "subgroups":
            # A subgroup built-in computes over the active invocations wherever it is called, as the other
            # languages' do, and the probes of reductions call it with some invocations inactive. So the probe turns
            # off the subgroup_uniformity diagnostic (WGSL, "Filterable Triggering Rules"), whose default severity,
            # error, requires subgroup-uniform control flow and lane arguments.
            lines.append("diagnostic(off, subgroup_uniformity);")
        return lines

    def keep(self, atomic: bool) -> str:
        # An atomic buffer element can only be used through atomic built-in functions.
        return "" if atomic else self.store("outp[i]")

    def discard(self, expression: str) -> str:
        # naga lowers a call statement of atomicLoad to no instruction; a phony assignment keeps the load.
        return f"_ = {expression};"

    def _globals(self, buffers: list[Buffer], parts: dict) -> list[str]:
        read_only = parts.get("read_only", ())
        lines = list(parts.get("prelude", ()))
        lines += [
            f"@group(0) @binding({binding}) var<storage, "
            + ("read" if name in read_only else "read_write")
            + f"> {name}: array<{kind}>;"
            for binding, (kind, name) in enumerate(buffers)
        ]
        return lines + list(parts.get("declarations", ()))

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = ", ".join(
            ["@builtin(global_invocation_id) id: vec3<u32>", *parts.get("params", ())]
        )
        lines = [
            *self._globals(buffers, parts),
            f"@compute @workgroup_size({parts.get('lanes', 1)}, 1, 1)",
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


DIALECTS: dict[str, Dialect] = {
    d.language: d
    for d in (
        CROSSGL,
        CUDA(),
        GLES(),
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

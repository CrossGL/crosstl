"""The dialect of each language a probe can be written in; CrossGL's own is in :mod:`.crossgl`."""

from __future__ import annotations

from .crossgl import DIALECT as CROSSGL
from .dialect import Buffer, Dialect, ProbeError
from .model import Type


class GLSL(Dialect):
    language = "glsl"
    suffix = ".comp"
    fragment_suffix = ".frag"

    def _globals(self, buffers: list[Buffer], parts: dict) -> list[str]:
        lines = [
            f"layout(std430, binding = {binding}) buffer B{binding} {{ {kind} {name}[]; }};"
            for binding, (kind, name) in enumerate(buffers)
        ]
        return lines + list(parts.get("declarations", ()))

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


class HLSL(Dialect):
    language = "hlsl"
    suffix = ".hlsl"
    rows_first = True
    entry_attributes: tuple[str, ...] = ("[numthreads(1, 1, 1)]",)
    fragment_attributes: tuple[str, ...] = ()

    def vector(self, scalar: str, size: int) -> str:
        return f"{scalar}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        return f"{scalar}{rows}x{columns}"

    def cast(self, kind: Type, value: str) -> str:
        return f"({self.type(kind)}){value}"

    def buffer(self, kind: str, name: str, binding: int) -> str:
        return f"RWStructuredBuffer<{kind}> {name} : register(u{binding});"

    def resource(self, spelling: str, name: str, register: str) -> str:
        return f"{spelling} {name} : register({register});"

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


class CUDA(Dialect):
    language = "cuda"
    suffix = ".cu"
    entry = "probe"

    def scalar(self, scalar: str) -> str:
        return "unsigned int" if scalar == "uint" else scalar

    def vector(self, scalar: str, size: int) -> str:
        if scalar == "bool":
            raise ProbeError("CUDA has no bool vectors")
        return f"{scalar}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        raise ProbeError("CUDA has no matrix types")

    def construct(self, kind: Type, parts: list[str]) -> str:
        return f"make_{self.type(kind)}({', '.join(parts)})"

    def cast(self, kind: Type, value: str) -> str:
        return f"({self.type(kind)}){value}"

    def out_argument(self, name: str) -> str:
        return f"&{name}"

    def inout_argument(self) -> str:
        return "&outp[i]"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = [f"{kind}* {name}" for kind, name in buffers]
        params += parts.get("params", ())
        lines = [
            *parts.get("prelude", ()),
            *parts.get("declarations", ()),
            f'extern "C" __global__ void probe({", ".join(params)}) {{',
            "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines + ["}"]) + "\n"


class HIP(CUDA):
    """HIP kernels are CUDA kernels that include the HIP runtime header, which hipcc does not add by itself."""

    language = "hip"
    suffix = ".hip"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        prelude = ["#include <hip/hip_runtime.h>", *parts.pop("prelude", ())]
        return super().kernel(buffers, body, prelude=prelude, **parts)


class Slang(HLSL):
    """Slang compiles HLSL kernels; the ``shader`` attribute marks the entry point, so slangc needs no ``-entry``."""

    language = "slang"
    suffix = ".slang"
    entry_attributes = ('[shader("compute")]', "[numthreads(1, 1, 1)]")
    fragment_attributes = ('[shader("fragment")]',)

    def buffer(self, kind: str, name: str, binding: int) -> str:
        # slangc assigns bindings in declaration order; a D3D register would only draw a warning for SPIR-V.
        return f"RWStructuredBuffer<{kind}> {name};"


class Mojo(Dialect):
    """A Mojo GPU kernel: buffers are pointer arguments and MAX's ``global_idx`` gives the invocation's index."""

    language = "mojo"
    suffix = ".mojo"
    entry = "probe"
    _SCALARS = {
        "float": "Float32",
        "double": "Float64",
        "int": "Int32",
        "uint": "UInt32",
        "bool": "Bool",
    }
    _DTYPES = {
        "float": "float32",
        "double": "float64",
        "int": "int32",
        "uint": "uint32",
        "bool": "bool",
    }

    def scalar(self, scalar: str) -> str:
        return self._SCALARS[scalar]

    def vector(self, scalar: str, size: int) -> str:
        return f"SIMD[DType.{self._DTYPES[scalar]}, {size}]"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        raise ProbeError("Mojo has no matrix types")

    def read(self, buffer: str, offset: int) -> str:
        return f"{buffer}[i + {offset}]"

    def literal(self, kind: Type) -> str:
        if kind.shape:
            raise ProbeError("no constant probe argument of a vector or matrix type")
        return {"bool": "False", "int": "0", "uint": "0"}.get(kind.scalar, "0.0")

    def declare(self, kind: Type, name: str, value: str = "") -> str:
        initial = f" = {value}" if value else ""
        return f"var {name}: {self.type(kind)}{initial}"

    def out_argument(self, name: str) -> str:
        raise ProbeError("Mojo has no output arguments")

    def to_int(self, condition: str) -> str:
        return f"Int32(1) if {condition} else Int32(0)"

    def first(self, vector: str) -> str:
        return f"{vector}[0]"

    def store(self, value: str) -> str:
        return f"outp[i] = {value}"

    def statement(self, expression: str) -> str:
        return expression

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        params = [
            f"{name}: UnsafePointer[{kind}, MutAnyOrigin]" for kind, name in buffers
        ]
        params += parts.get("params", ())
        lines = [
            "from std.bit import *",
            "from std.math import *",
            "from max.gpu import global_idx",
            *parts.get("prelude", ()),
            "",
            *parts.get("declarations", ()),
            "",
            f"def probe({', '.join(params)}):",
            "    var i: Int = global_idx.x",
        ]
        lines += [f"    {line}" for line in body]
        return "\n".join(lines) + "\n"


class Rust(Dialect):
    """A rust-gpu compute entry point: buffers are slice parameters and glam supplies the vector and matrix types.

    A call spelled ``.name`` is a method of its first argument, and one spelled ``unsafe name`` sits in an ``unsafe``
    block.
    """

    language = "rust"
    suffix = ".rs"
    entry = "probe"
    _SCALARS = {
        "float": "f32",
        "double": "f64",
        "int": "i32",
        "uint": "u32",
        "bool": "bool",
    }
    _GLAM = {"float": "", "double": "D", "int": "I", "uint": "U", "bool": "B"}

    def scalar(self, scalar: str) -> str:
        return self._SCALARS[scalar]

    def vector(self, scalar: str, size: int) -> str:
        return f"{self._GLAM[scalar]}Vec{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        if scalar not in ("float", "double") or columns != rows:
            raise ProbeError("glam has square float matrices only")
        return f"{self._GLAM[scalar]}Mat{columns}"

    def read(self, buffer: str, offset: int) -> str:
        return f"{buffer}[i + {offset}]"

    def construct(self, kind: Type, parts: list[str]) -> str:
        build = "from_cols" if len(kind.shape) == 2 else "new"
        return f"{self.type(kind)}::{build}({', '.join(parts)})"

    def cast(self, kind: Type, value: str) -> str:
        if kind.shape:
            return f"{value}.as_{self.type(kind).lower()}()"
        return f"({value}) as {self.type(kind)}"

    def literal(self, kind: Type) -> str:
        if kind.shape:
            raise ProbeError("no constant probe argument of a vector or matrix type")
        return {"bool": "false", "int": "0", "uint": "0"}.get(kind.scalar, "0.0")

    def declare(self, kind: Type, name: str, value: str = "") -> str:
        if not value:
            return f"let mut {name}: {self.type(kind)} = Default::default();"
        return f"let {name}: {self.type(kind)} = {value};"

    def call(self, name: str, arguments: list[str]) -> str:
        unsafe = name.startswith("unsafe ")
        name = name.removeprefix("unsafe ")
        if name.startswith("."):
            receiver, *rest = arguments
            call = f"{receiver}{name}({', '.join(rest)})"
        else:
            call = f"{name}({', '.join(arguments)})"
        return f"unsafe {{ {call} }}" if unsafe else call

    def out_argument(self, name: str) -> str:
        return f"&mut {name}"

    def inout_argument(self) -> str:
        return "&mut outp[i]"

    def to_int(self, condition: str) -> str:
        return f"if {condition} {{ 1 }} else {{ 0 }}"

    def part(self, matrix: str) -> str:
        return f"{matrix}.x_axis"

    def buffer_element(self, kind: Type, atomic: bool) -> str:
        # A storage buffer of 3-vectors needs a 16-byte stride, which glam's 12-byte vectors lack.
        if kind.shape == (3,):
            return self.type(Type(kind.scalar, (4,)))
        return self.type(kind)

    def stored(self, kind: Type, value: str) -> str:
        if kind.shape == (3,):
            return f"{value}.extend({self.literal(Type(kind.scalar))})"
        return value

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        entry = (
            "#[spirv(compute(threads(1)))]",
            "#[spirv(global_invocation_id)] id: UVec3",
        )
        return self._entry(entry, buffers, ["let i = id.x as usize;", *body], parts)

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        parts = {**parts, "params": [*parts.get("params", ()), "color: &mut Vec4"]}
        body = ["let i = 0usize;", *body, "*color = Vec4::ZERO;"]
        return self._entry(("#[spirv(fragment)]",), buffers, body, parts)

    def _entry(
        self,
        entry: tuple[str, ...],
        buffers: list[Buffer],
        body: list[str],
        parts: dict,
    ) -> str:
        """An entry point: ``entry`` is its attribute and any built-in parameters, ahead of the buffers."""
        attribute, *params = entry
        for binding, (kind, name) in enumerate(buffers):
            access = "&mut " if name == "outp" else "&"
            params.append(
                f"#[spirv(storage_buffer, descriptor_set = 0, binding = {binding})] {name}: {access}[{kind}]"
            )
        params += parts.get("params", ())
        prelude = list(parts.get("prelude", ()))
        # Crate attributes (``#![feature(f16)]``) have to precede every item.
        attributes = [line for line in prelude if line.startswith("#!")]
        declarations = list(parts.get("declarations", ()))
        if declarations:
            declarations.append("")
        lines = [
            "#![no_std]",
            *attributes,
            "use spirv_std::arch::*;",
            "use spirv_std::glam::*;",
            "use spirv_std::memory::{Scope, Semantics};",
            "use spirv_std::num_traits::Float;",
            "use spirv_std::spirv;",
            *(line for line in prelude if not line.startswith("#!")),
            "",
            *declarations,
            attribute,
            "pub fn probe(",
            *(f"    {param}," for param in params),
            ") {",
        ]
        lines += [f"    {line}" for line in body]
        lines.append("}")
        return "\n".join(lines) + "\n"


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

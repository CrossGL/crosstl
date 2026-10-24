"""The dialects of the compute-kernel languages a probe can be written in: CUDA, HIP, OpenCL C, Mojo and Rust."""

from __future__ import annotations

import re

from .dialect import Buffer, Dialect, Handle, ProbeError
from .model import OPENCL_ATOMICS as ATOMIC_TYPES
from .model import OPENCL_ENUMERATIONS as ENUMERATIONS
from .model import Type


class CUDA(Dialect):
    language = "cuda"
    suffix = ".cu"
    entry = "probe"
    wide_integers = True
    string = "const char*"
    _SCALARS = {
        "uint": "unsigned int",
        "int64": "long long",
        "uint64": "unsigned long long",
    }
    _VECTORS = {"int64": "longlong", "uint64": "ulonglong"}

    def scalar(self, scalar: str) -> str:
        return self._SCALARS.get(scalar, scalar)

    def vector(self, scalar: str, size: int) -> str:
        if scalar == "bool":
            raise ProbeError("CUDA has no bool vectors")
        return f"{self._VECTORS.get(scalar, scalar)}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        raise ProbeError("CUDA has no matrix types")

    def construct(self, kind: Type, parts: list[str]) -> str:
        return f"make_{self.type(kind)}({', '.join(parts)})"

    def cast(self, kind: Type, value: str) -> str:
        return f"({self.type(kind)}){value}"

    def out_argument(self, name: str) -> str:
        return f"&{name}"

    def address(self, name: str) -> str:
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


# The OpenCL C handle types a built-in makes, and the call a probe makes one with. An event_t is zero where no copy
# made it (OpenCL C 3.0, section 6.15.11); the others come from the device-side enqueue functions (section 6.15.16).
_HANDLES = {
    "event_t": "0",
    "queue_t": "get_default_queue()",
    "clk_event_t": "create_user_event()",
    "ndrange_t": "ndrange_1D(1)",
}
# A reservation of packets in a pipe, which the kernel takes as an argument (section 6.15.17).
_PIPE = "read_only pipe int"
# A pointer into one named address space, a generic pointer, and a constant array, as the header spells them.
_SPACE_POINTER = re.compile(r"(?:const )?(__local|__global) (\w+) ?\*")
_POINTER = re.compile(r"(?:const )?(\w+) ?\*")
_ARRAY = re.compile(r"const (\w+)\[(\d+)\]")


class OpenCL(Dialect):
    """An OpenCL C kernel: buffers are ``__global`` pointer arguments and ``get_global_id`` gives the invocation's
    index. OpenCL C has C's casts and vector literals rather than constructors, and no references, so an out argument
    is the address of a private local."""

    language = "opencl"
    suffix = ".cl"
    entry = "probe"
    wide_integers = True
    narrow_scalars = True
    byte_scalars = True
    _SCALARS = {
        "int8": "char",
        "uint8": "uchar",
        "int16": "short",
        "uint16": "ushort",
        "int64": "long",
        "uint64": "ulong",
    }
    _HALF = re.compile(r"\bhalf\d*\b")
    _FP16 = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
    # An image or a pipe is a kernel argument, declared with its access qualifier.
    _IMAGE = re.compile(r"(?:read_only|write_only|read_write) (?:image|pipe)")
    # A sampler every image read accepts: integer and floating-point coordinates, and integer image formats.
    _SAMPLER = (
        "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST"
    )

    def type(self, kind: Type) -> str:
        if kind.opaque in ENUMERATIONS:
            return kind.opaque
        if kind.opaque in ATOMIC_TYPES:
            return kind.opaque
        return super().type(kind)

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        if spelling == "sampler_t":
            return f"__constant sampler_t {name} = {self._SAMPLER};"
        return f"{spelling} {name};"

    def scalar(self, scalar: str) -> str:
        return self._SCALARS.get(scalar, scalar)

    def vector(self, scalar: str, size: int) -> str:
        if scalar == "bool":
            raise ProbeError("OpenCL C has no bool vectors")
        return f"{self.scalar(scalar)}{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        raise ProbeError("OpenCL C has no matrix types")

    def literal(self, kind: Type) -> str:
        if kind.opaque in ENUMERATIONS:
            return ENUMERATIONS[kind.opaque]
        if kind.scalar == "float" and not kind.shape:
            return "0.0f"
        return super().literal(kind)

    def constant(self, kind: Type) -> str:
        if kind.opaque in ENUMERATIONS:
            return ENUMERATIONS[kind.opaque]
        return super().constant(kind)

    def construct(self, kind: Type, parts: list[str]) -> str:
        return f"({self.type(kind)})({', '.join(parts)})"

    def cast(self, kind: Type, value: str) -> str:
        # A C cast converts a scalar; OpenCL C converts a vector only through convert_<type>.
        if kind.shape:
            return f"convert_{self.type(kind)}({value})"
        return f"({self.type(kind)}){value}"

    def positive(self, value: str) -> str:
        return f"({value} > 0.0f)"

    def out_argument(self, name: str) -> str:
        return f"&{name}"

    def address(self, name: str) -> str:
        return f"&{name}"

    def inout_argument(self) -> str:
        return "&outp[i]"

    def workgroup(self, kind: Type, name: str) -> str:
        count = f"[{kind.count}]" if kind.count else ""
        return f"__local {self.type(kind.element)} {name}{count};"

    def holding(self, opaque: str) -> str:
        return opaque if opaque in _HANDLES else ""

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        # An enumeration is an unsigned integer: get_fence returns the cl_mem_fence_flags of a pointer.
        return (Type("uint"), "(uint){}") if opaque in ENUMERATIONS else None

    def handle(self, kind: Type, name: str) -> Handle | None:
        """A handle the built-in that makes it returns; an image or a pipe the kernel takes; a ``__local`` pointer
        to a work-group array, or a ``__global`` one into the output buffer; a generic pointer to a handle; and a
        constant array of ones."""
        spelling = kind.opaque
        if kind.scalar or not spelling:
            return None
        if spelling in _HANDLES:
            return Handle(name, (f"{spelling} {name} = {_HANDLES[spelling]};",))
        if spelling == "reserve_id_t":
            pipe = f"{name}_pipe"
            line = f"reserve_id_t {name} = reserve_read_pipe({pipe}, 1u);"
            return Handle(name, (line,), (f"{_PIPE} {pipe};",))
        if self._IMAGE.match(spelling):
            return Handle(name, declarations=(f"{spelling} {name};",))
        space = _SPACE_POINTER.fullmatch(spelling)
        if space and space.group(1) == "__local":
            return Handle(name, declarations=(f"__local {space.group(2)} {name}[1];",))
        if space:
            return Handle(f"({spelling})outp")
        pointer = _POINTER.fullmatch(spelling)
        if pointer and pointer.group(1) == "void":
            return Handle("outp")
        if pointer and pointer.group(1) in _HANDLES:
            pointee = pointer.group(1)
            return Handle(f"&{name}", (f"{pointee} {name} = {_HANDLES[pointee]};",))
        array = _ARRAY.fullmatch(spelling)
        if array:
            element, count = array.group(1), int(array.group(2))
            ones = ", ".join(["1"] * count)
            return Handle(name, (f"{element} {name}[{count}] = {{{ones}}};",))
        return None

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        # OpenCL C declares a __local variable in the kernel's own scope, not at program scope.
        declarations = list(parts.get("declarations", ()))
        shared = [line for line in declarations if line.startswith("__local ")]
        images = [line for line in declarations if self._IMAGE.match(line)]
        params = [f"__global {kind}* {name}" for kind, name in buffers]
        params += [line.rstrip(";") for line in images]
        params += parts.get("params", ())
        lines = [
            *parts.get("prelude", ()),
            *(line for line in declarations if line not in shared + images),
            f"__kernel void probe({', '.join(params)}) {{",
            "    uint i = get_global_id(0);",
        ]
        lines += [f"    {line}" for line in [*shared, *body]]
        source = "\n".join(lines + ["}"]) + "\n"
        if self._HALF.search(source) and self._FP16 not in source:
            source = f"{self._FP16}\n{source}"
        return source


class Mojo(Dialect):
    """A Mojo GPU kernel: buffers are pointer arguments and MAX's ``global_idx`` gives the invocation's index."""

    language = "mojo"
    suffix = ".mojo"
    one = "1"
    entry = "probe"
    # Mojo's 64-bit integer scalar is the word-sized Int, which its integer functions take.
    wide_integers = True
    _SCALARS = {
        "float": "Float32",
        "double": "Float64",
        "int": "Int32",
        "uint": "UInt32",
        "bool": "Bool",
        "int64": "Int",
    }
    _DTYPES = {
        "float": "float32",
        "double": "float64",
        "int": "int32",
        "uint": "uint32",
        "bool": "bool",
        "int64": "int64",
    }
    # A pair of values, as Mojo's frexp, modf and divmod return; the probe reads the first.
    _PAIR = re.compile(r"Tuple\[(\w+), \1\]|StaticTuple\[(\w+), 2\]")

    def scalar(self, scalar: str) -> str:
        return self._SCALARS[scalar]

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        pair = self._PAIR.fullmatch(opaque)
        element = pair and (pair.group(1) or pair.group(2))
        scalars = {name: scalar for scalar, name in self._SCALARS.items()}
        if element not in scalars:
            return None
        return Type(scalars[element]), "{}[0]"

    def vector(self, scalar: str, size: int) -> str:
        return f"SIMD[DType.{self._DTYPES[scalar]}, {size}]"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        raise ProbeError("Mojo has no matrix types")

    def read(self, buffer: str, offset: int) -> str:
        return f"{buffer}[i + {offset}]"

    def literal(self, kind: Type) -> str:
        if kind.shape:
            raise ProbeError("no constant probe argument of a vector or matrix type")
        return {"bool": "False", "int": "0", "uint": "0", "int64": "0"}.get(
            kind.scalar, "0.0"
        )

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
    one = "1"
    entry = "probe"
    # rustgpu.CAPABILITIES enables Int16 and Int64 for every probe.
    wide_integers = True
    pointer_sized = True
    narrow_scalars = True
    _SCALARS = {
        "float": "f32",
        "double": "f64",
        "int": "i32",
        "uint": "u32",
        "bool": "bool",
        "int16": "i16",
        "uint16": "u16",
        "int64": "i64",
        "uint64": "u64",
        "usize": "usize",
    }
    # spirv-std's glam has 32-bit integer vectors only (features/rust.py, _GLAM_FEATURES).
    _GLAM = {"float": "", "double": "D", "int": "I", "uint": "U", "bool": "B"}
    _INTEGERS = frozenset(
        {"int", "uint", "int16", "uint16", "int64", "uint64", "usize"}
    )
    # A tuple result, read through its first field, and an optional one, read as its value or the type's default.
    _HELD = re.compile(r"\((\w+), [^()]+\)|Option<(\w+)>")
    # Results no buffer element holds, read as a type one does: a float's category as its discriminant, and glam's
    # 16-byte aligned 3-vector as its plain one.
    _CONVERTED = {
        "FpCategory": (Type("uint"), "({}) as u32"),
        "Vec3A": (Type("float", (3,)), "Vec3::from({})"),
    }

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        if opaque in self._CONVERTED:
            return self._CONVERTED[opaque]
        held = self._HELD.fullmatch(opaque)
        if not held:
            return None
        spelled = held.group(1) or held.group(2)
        scalars = {name: scalar for scalar, name in self._SCALARS.items()}
        vector = re.fullmatch(r"([DIUB]?)Vec([234])", spelled)
        if spelled in scalars:
            kind = Type(scalars[spelled])
        elif vector:
            prefixes = {prefix: scalar for scalar, prefix in self._GLAM.items()}
            kind = Type(prefixes[vector.group(1)], (int(vector.group(2)),))
        else:
            return None
        return kind, "{}.0" if held.group(1) else "{}.unwrap_or_default()"

    def type(self, kind: Type) -> str:
        if kind.count:
            return f"[{self.type(kind.element)}; {kind.count}]"
        return super().type(kind)

    def scalar(self, scalar: str) -> str:
        if scalar not in self._SCALARS:
            raise ProbeError(f"no {scalar} probe for rust yet")
        return self._SCALARS[scalar]

    def vector(self, scalar: str, size: int) -> str:
        if scalar not in self._GLAM:
            raise ProbeError(f"spirv-std's glam has no {scalar} vectors")
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

    def array(self, kind: Type, parts: list[str]) -> str:
        return f"[{', '.join(parts)}]"

    def literal(self, kind: Type) -> str:
        if kind.shape:
            raise ProbeError("no constant probe argument of a vector or matrix type")
        if kind.scalar == "bool":
            return "false"
        return "0" if kind.scalar in self._INTEGERS else "0.0"

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

    def geometry(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        entry = (
            "#[spirv(geometry(input_points = 1, output_points = 1, output_vertices = 1))]",
        )
        return self._entry(entry, buffers, ["let i = 0usize;", *body], parts)

    # Hit shaders take the ray payload and the triangle's barycentrics, as the other languages' hit probes do, and a
    # callable shader takes the data it is called with.
    _PAYLOAD = ("pub struct Payload {", "    pub color: Vec4,", "}")
    _HIT_PARAMETERS = (
        "#[spirv(incoming_ray_payload)] payload: &mut Payload",
        "#[spirv(hit_attribute)] attributes: &mut Vec2",
    )

    def raytracing(
        self, stage: str, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        entry = (f"#[spirv({stage.replace('-', '_')})]",)
        if stage in ("any-hit", "closest-hit", "callable"):
            entry += (
                self._HIT_PARAMETERS
                if stage != "callable"
                else (f"#[spirv(incoming_callable_data)] {self.payload}: &mut Payload",)
            )
            declarations = [*self._PAYLOAD, *parts.get("declarations", ())]
            parts = {**parts, "declarations": declarations}
        return self._entry(entry, buffers, ["let i = 0usize;", *body], parts)

    # A mesh shader that writes one triangle, and a task shader with a payload for the mesh workgroups it launches.
    # rustgpu.MESH_TARGET compiles both for SPIR-V 1.4 or later, which SPV_EXT_mesh_shader needs.
    _MESH = (
        "#[spirv(mesh_ext(threads(1), output_vertices = 3, output_primitives_ext = 1, output_triangles_ext))]",
        "#[spirv(position)] positions: &mut [Vec4; 3]",
        "#[spirv(primitive_triangle_indices_ext)] indices: &mut [UVec3; 1]",
    )
    _TRIANGLE = (
        "positions[0] = Vec4::new(0.0, 0.0, 0.0, 1.0);",
        "positions[1] = Vec4::new(1.0, 0.0, 0.0, 1.0);",
        "positions[2] = Vec4::new(0.0, 1.0, 0.0, 1.0);",
        "indices[0] = UVec3::new(0, 1, 2);",
    )
    _TASK_PAYLOAD = ("pub struct Payload {", "    pub value: u32,", "}")

    def mesh(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        body = ["let i = 0usize;", *body, *self._TRIANGLE]
        return self._entry(self._MESH, buffers, body, parts)

    def task(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        entry = (
            "#[spirv(task_ext(threads(1)))]",
            f"#[spirv(task_payload_workgroup_ext)] {self.payload}: &mut Payload",
        )
        declarations = [*self._TASK_PAYLOAD, *parts.get("declarations", ())]
        parts = {**parts, "declarations": declarations}
        return self._entry(entry, buffers, ["let i = 0usize;", *body], parts)

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

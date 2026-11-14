"""The dialects of the compute-kernel languages a probe can be written in: CUDA, HIP, Mojo and Rust. OpenCL C is in
:mod:`.opencl`."""

from __future__ import annotations

import re

from .dialect import Buffer, Dialect, Handle, ProbeError
from .model import GENERIC_POINTER, Type


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

    def workgroup(self, kind: Type, name: str) -> str:
        count = f"[{kind.count}]" if kind.count else ""
        return f"__shared__ {self.type(kind.element)} {name}{count};"

    def constant_memory(self, kind: Type, name: str) -> str:
        return f"__constant__ {self.type(kind)} {name};"

    def holding(self, opaque: str) -> str:
        # A generic pointer, which the address space conversions return.
        return "void*" if opaque == GENERIC_POINTER else ""

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
        "uint64": "UInt64",
    }
    _DTYPES = {
        "float": "float32",
        "double": "float64",
        "int": "int32",
        "uint": "uint32",
        "bool": "bool",
        "int64": "int64",
        "uint64": "uint64",
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
        return {
            "bool": "False",
            "int": "0",
            "uint": "0",
            "int64": "0",
            "uint64": "0",
        }.get(kind.scalar, "0.0")

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

    def enable(self, extension: str) -> list[str]:
        # A Mojo feature's extension is what the probe imports, one name or more: the module ``warp`` for
        # max.gpu.primitives.warp, the function ``barrier`` for max.gpu.sync.barrier, or the types of
        # ``std.atomic.Atomic std.atomic.Ordering``.
        lines = []
        for imported in extension.split():
            package, _, name = imported.rpartition(".")
            lines.append(f"from {package} import {name}")
        return lines

    def inout_argument(self) -> str:
        # An atomic operation takes a pointer to the output buffer's element.
        return "outp + i"

    def discard(self, expression: str) -> str:
        return f"_ = {expression}"

    def address(self, name: str) -> str:
        # A ``mut`` argument is passed by reference without an operator.
        return name

    def active(self, body: list[str], skipped: int) -> list[str]:
        return [f"if i % {skipped} != 0:", *(f"    {line}" for line in body)]

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
    # spirv-std's glam has 32-bit integer vectors only (features/rust.py, GLAM_FEATURES).
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
        "Mat3A": (Type("float", (3, 3)), "Mat3::from({})"),
        "Quat": (Type("float", (4,)), "Vec4::from({})"),
        "DQuat": (Type("double", (4,)), "DVec4::from({})"),
        # A ray query's intersection kinds as their discriminants and its ray flags as their bits; an object-to-world
        # matrix as its translation, and a triangle's vertex positions as its first vertex (features/ray_query.py).
        "CandidateIntersection": (Type("uint"), "({}) as u32"),
        "CommittedIntersection": (Type("uint"), "({}) as u32"),
        "RayFlags": (Type("uint"), "({}).bits()"),
        "Matrix4x3": (Type("float", (3,)), "({}).w_axis"),
        "[Vec3; 3]": (Type("float", (3,)), "({})[0]"),
    }
    # Arguments of glam's types that no buffer holds, made from the values glam converts them from. A projective depth
    # comparison's coordinate and depth reference come from one vector in GLSL's order, the depth reference third
    # (features/spirv_std.py, PROJECTIVE_DEPTH); the argument is two arguments.
    _MADE = {
        "Vec3A": (Type("float", (3,)), "Vec3A::from({})"),
        "Mat3A": (Type("float", (3, 3)), "Mat3A::from({})"),
        "Quat": (Type("float", (4,)), "Quat::from_vec4({})"),
        "DQuat": (Type("double", (4,)), "DQuat::from_vec4({})"),
        "ProjectiveDepth": (Type("float", (4,)), "{0}.xyw(), {0}.z"),
        # A byte-address buffer's index, four times a word index (features/spirv_std.py, BYTE_INDEX).
        "ByteIndex": (Type("uint"), "{} * 4"),
        # A ray's flags keep the word's bits, as HLSL and GLSL pass them: from_bits_truncate would mask them, and the
        # mask would join the semantic id of the call that takes them.
        "RayFlags": (
            Type("uint"),
            "spirv_std::ray_tracing::RayFlags::from_bits_retain({})",
        ),
    }
    # spirv_std makes a ray query with a macro, which declares a reference to it.
    own_objects = {"RayQuery": ("spirv_std::ray_query!(let mut {name});",)}
    # A closure parameter's type, as features/rust.py keeps its bound: one argument and a result of the same type.
    _CLOSURE = re.compile(r"Fn(?:Mut|Once)?\((\w+)\) -> \1")

    def handle(self, kind: Type, name: str) -> Handle | None:
        closure = self._CLOSURE.fullmatch(kind.opaque)
        numeric = {
            spelled for scalar, spelled in self._SCALARS.items() if scalar != "bool"
        }
        if kind.opaque == "EulerRot":
            # A probe converts Euler angles in the order x, then y, then z.
            return Handle("EulerRot::XYZ")
        if not closure or closure.group(1) not in numeric:
            return None
        # The closure doubles its argument, so a translation that drops the call computes something else.
        return Handle(f"|x: {closure.group(1)}| x + x")

    def making(self, opaque: str) -> tuple[Type, str] | None:
        return self._MADE.get(opaque)

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        if opaque in self._CONVERTED:
            return self._CONVERTED[opaque]
        held = self._HELD.fullmatch(opaque)
        if not held:
            return None
        spelled = held.group(1) or held.group(2)
        scalars = {name: scalar for scalar, name in self._SCALARS.items()}
        glam = re.fullmatch(r"([DIUB]?)(Vec|Mat)([234])", spelled)
        if spelled in scalars:
            kind = Type(scalars[spelled])
        elif glam:
            prefixes = {prefix: scalar for scalar, prefix in self._GLAM.items()}
            size = int(glam.group(3))
            shape = (size, size) if glam.group(2) == "Mat" else (size,)
            kind = Type(prefixes[glam.group(1)], shape)
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

    def constant(self, kind: Type) -> str:
        # glam's zero vectors are constants.
        if len(kind.shape) == 1:
            return f"{self.type(kind)}::ZERO"
        return super().constant(kind)

    # rust-gpu takes a shader's resources as parameters of its entry point, which _entry moves these declarations to.
    _RESOURCE = "#[spirv("
    # The spirv_std items a resource's type names.
    _RESOURCE_IMPORTS = (
        ("Image!", "use spirv_std::Image;"),
        ("Sampler", "use spirv_std::Sampler;"),
        ("SampledImage<", "use spirv_std::image::SampledImage;"),
        ("AccelerationStructure", "use spirv_std::ray_tracing::AccelerationStructure;"),
    )
    # What a probe needs for a spirv_std macro it calls: ray_query! names RayQuery unqualified, and expands to inline
    # assembly, which the crate has to enable for the SPIR-V target.
    _MACRO_PRELUDES = (
        (
            "spirv_std::ray_query!",
            (
                "#![feature(asm_experimental_arch)]",
                "use spirv_std::ray_tracing::RayQuery;",
            ),
        ),
    )

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        attributes = [qualifier] if qualifier else []
        attributes += ["descriptor_set = 0", f"binding = {binding}"]
        return f"{self._RESOURCE}{', '.join(attributes)})] {name}: &{spelling}"

    # A method takes a sampler by value, which the entry point takes a reference to, and a byte-address buffer is made
    # from the storage buffer of words the entry point takes.
    _UNIFORM_ARGUMENTS = {
        "Sampler": "*{}",
        "ByteAddressableBuffer<&[u32]>": (
            "spirv_std::ByteAddressableBuffer::from_slice({})"
        ),
        "ByteAddressableBuffer<&mut [u32]>": (
            "spirv_std::ByteAddressableBuffer::from_mut_slice({})"
        ),
    }

    def uniform_argument(self, opaque: str, name: str) -> str:
        return self._UNIFORM_ARGUMENTS.get(opaque, "{}").format(name)

    def declare(self, kind: Type, name: str, value: str = "") -> str:
        if not value:
            return f"let mut {name}: {self.type(kind)} = Default::default();"
        return f"let {name}: {self.type(kind)} = {value};"

    def retain(self, kind: Type, name: str, statement: str) -> list[str]:
        # rust-gpu removes values nothing uses. A baseline that only declared a boolean argument would lose the
        # comparison that makes it, and the comparison would then join the id of every call that takes one, as
        # OpFOrdGreaterThan did subgroup_all's. Each component guards the store instead.
        parts = (
            [f"{name}.{c}" for c in "xyzw"[: kind.shape[0]]] if kind.shape else [name]
        )
        return [f"if {part} {{ {statement} }}" for part in parts]

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

    def address(self, name: str) -> str:
        return f"&{name}"

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

    def active(self, body: list[str], skipped: int) -> list[str]:
        return [f"if i % {skipped} != 0 {{", *(f"    {line}" for line in body), "}"]

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        entry = (
            f"#[spirv(compute(threads({parts.get('lanes', 1)})))]",
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
        declarations = list(parts.get("declarations", ()))
        resources = [line for line in declarations if line.startswith(self._RESOURCE)]
        declarations = [line for line in declarations if line not in resources]
        params += resources
        params += parts.get("params", ())
        prelude = list(parts.get("prelude", ()))
        for macro, lines in self._MACRO_PRELUDES:
            if any(macro in line for line in body):
                prelude += lines
        prelude += [
            line
            for name, line in self._RESOURCE_IMPORTS
            if any(name in resource.partition(":")[2] for resource in resources)
        ]
        # Crate attributes (``#![feature(f16)]``) have to precede every item.
        attributes = [line for line in prelude if line.startswith("#!")]
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

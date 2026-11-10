"""The OpenCL C dialect. CrossTL imports OpenCL C but has no OpenCL C generator, so only input probes use it."""

from __future__ import annotations

import re

from .dialect import Buffer, Dialect, Handle, ProbeError
from .model import OPENCL_ATOMICS as ATOMIC_TYPES
from .model import OPENCL_ENUMERATIONS as ENUMERATIONS
from .model import Type

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
    _GROUP_SIZE = "__attribute__((reqd_work_group_size(1, 1, 1)))"
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
            # Like the other languages' probes, the kernel fixes its work-group size at one invocation.
            f"__kernel {self._GROUP_SIZE} void probe({', '.join(params)}) {{",
            "    uint i = get_global_id(0);",
        ]
        lines += [f"    {line}" for line in [*shared, *body]]
        source = "\n".join(lines + ["}"]) + "\n"
        if self._HALF.search(source) and self._FP16 not in source:
            source = f"{self._FP16}\n{source}"
        return source

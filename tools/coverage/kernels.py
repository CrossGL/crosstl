"""The dialects of the compute-kernel languages a probe can be written in: CUDA, HIP and Mojo. Rust is in :mod:`.rust`
and OpenCL C in :mod:`.opencl`."""

from __future__ import annotations

import re

from .dialect import Buffer, Dialect, ProbeError
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

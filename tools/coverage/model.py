"""Value types shared by the feature lists, probes and report."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SCALARS = (
    "bool",
    "int",
    "uint",
    "float",
    "double",
    "int64",
    "uint64",
    "half",
    "int16",
    "uint16",
    "int8",
    "uint8",
    "int8x4_packed",
    "uint8x4_packed",
    "usize",
)
# Scalars a probe reads from and stores as 32-bit buffer elements: the 64-bit integers, the 16-bit and 8-bit scalars,
# the 32-bit words that pack four 8-bit integers (HLSL shader model 6.6), and Rust's pointer-sized unsigned integer,
# which rust-gpu's SPIR-V targets make 32 bits wide.
WIDE_INTEGERS = frozenset({"int64", "uint64"})
NARROW_SCALARS = frozenset({"half", "int16", "uint16"})
BYTES = frozenset({"int8", "uint8"})
PACKED_BYTES = frozenset({"int8x4_packed", "uint8x4_packed"})
POINTER_SIZED = frozenset({"usize"})
# Overload selection prefers a signature without these when a function has one.
OTHER_WIDTHS = frozenset({"double"}) | WIDE_INTEGERS | NARROW_SCALARS | BYTES
OUTCOMES = ("supported", "refused", "wrong", "untested")
LANGUAGES = (
    "cuda",
    "glsl",
    "hip",
    "hlsl",
    "metal",
    "mojo",
    "opencl",
    "rust",
    "slang",
    "spirv",
    "webgl",
    "wgsl",
)
DIRECTIONS = ("output", "input")
# The language an imported probe is emitted to and validated in when CrossTL has no generator for its own: HLSL, the
# target whose output row covers the most of what OpenCL C kernels call.
ROUND_TRIPS = {"opencl": "hlsl"}
# The ray-tracing stages a probe can run in, each with the name HLSL's ``[shader(...)]`` attribute and slangc's
# ``-stage`` give it. A ray-tracing intrinsic is probed in a closest-hit shader, which allows most of them, unless only
# one other stage has it (DirectX Raytracing functional specification, "Intrinsics", and HLSL shader execution
# reordering).
RAYTRACING_STAGES = {
    "ray-generation": "raygeneration",
    "intersection": "intersection",
    "any-hit": "anyhit",
    "closest-hit": "closesthit",
    "miss": "miss",
    "callable": "callable",
}
_RAYTRACING_ONLY = {
    "AcceptHitAndEndSearch": "any-hit",
    "IgnoreHit": "any-hit",
    "ReportHit": "intersection",
    "ReportHitOptix": "intersection",
    "MaybeReorderThread": "ray-generation",
    "ReorderThread": "ray-generation",
    "dx::MaybeReorderThread": "ray-generation",
}


def raytracing_stage(name: str) -> str:
    """The ray-tracing stage that probes the intrinsic ``name``."""
    return _RAYTRACING_ONLY.get(name, "closest-hit")


@dataclass(frozen=True)
class Type:
    """A value type: a scalar, a vector ``(n,)`` or a matrix ``(columns, rows)``, or an array of ``count`` of them.

    Any other kind of type, such as a sampler or an image, is carried as ``opaque`` with its upstream spelling. A
    probe can pass only the samplers, images and objects that :mod:`.objects` declares.
    """

    scalar: str
    shape: tuple[int, ...] = ()
    opaque: str = ""
    count: int = 0

    def __post_init__(self):
        if not self.opaque and self.scalar not in SCALARS:
            raise ValueError(f"unknown scalar type {self.scalar!r}")
        if len(self.shape) > 2 or any(not 2 <= n <= 4 for n in self.shape):
            raise ValueError(f"unsupported shape {self.shape!r}")
        if self.count < 0 or (self.count and self.opaque):
            raise ValueError(f"unsupported array of {self.count} {self.opaque!r}")

    @property
    def element(self) -> Type:
        """The type of one element of an array, or the type itself."""
        return Type(self.scalar, self.shape, self.opaque)

    @classmethod
    def other(cls, spelling: str) -> Type:
        return cls("", (), spelling)

    @property
    def components(self) -> int:
        count = 1
        for n in self.shape:
            count *= n
        return count

    def with_scalar(self, scalar: str) -> Type:
        return Type(scalar, self.shape)


VOID = Type.other("void")
# The OpenCL C enumerated types of the synchronization, atomic and event-profiling functions, whose arguments are named
# constants, and the constant a probe passes. They are kept as ``uint`` types with the enumeration as ``opaque``.
OPENCL_ENUMERATIONS = {
    "cl_mem_fence_flags": "CLK_LOCAL_MEM_FENCE",
    "memory_scope": "memory_scope_work_group",
    "memory_order": "memory_order_relaxed",
    "clk_profiling_info": "CLK_PROFILING_COMMAND_EXEC_TIME",
}
# The C11 atomic types of OpenCL C (OpenCL C 3.0, section 6.15.12.6), by the type of the value each holds. They are
# kept as that type with the atomic type as ``opaque``; a probe passes the address of an element of an output buffer
# of the atomic type.
OPENCL_ATOMICS = {
    "atomic_int": "int",
    "atomic_uint": "uint",
    "atomic_long": "int64",
    "atomic_ulong": "uint64",
    "atomic_float": "float",
    "atomic_double": "double",
    "atomic_half": "half",
    "atomic_flag": "int",
}
# A constant string argument, such as the tag a C library ``nan`` takes; probes pass the empty string.
STRING = Type.other("string")
_COOP_VECTOR = re.compile(r"CoopVec<(\w+), ([2-4])>")
# The length a probe gives a cooperative vector whose length is a generic size.
COOP_VECTOR_LENGTH = 4
# The column count ``k`` that Slang's packed cooperative-vector multiplies take as a constant. For the unpacked 32-bit
# floats every probe passes, it must equal the input vector's length (hlsl.meta.slang, coopVecMatMulPacked).
COOP_COLUMNS = Type("int", opaque="CoopVecColumns")
# The constant a probe passes for each enumeration parameter of Slang's cooperative-vector functions: a vector's and a
# matrix's elements are the 32-bit floats every probe fills them with, in row-major order, and a coherent load is
# coherent across the device, the scope coopVecLoadCoherent defaults to.
COOP_ENUMERATORS = {
    "CoopVecComponentType": "CoopVecComponentType.Float32",
    "CoopVecMatrixLayout": "CoopVecMatrixLayout.RowMajor",
    "MemoryScope": "MemoryScope.Device",
}


def coop_vector(scalar: str, length: int) -> Type:
    """Slang's cooperative vector of ``length`` ``scalar`` elements: an opaque type that keeps its element type and
    length as ``scalar`` and ``shape``, so a probe can fill one from the input buffers.
    """
    return Type(scalar, (length,), f"CoopVec<{scalar}, {length}>")


def coop_literal(kind: Type) -> str:
    """The constant a probe passes for a cooperative-vector enumeration or column count ``kind``, or an empty
    string."""
    if kind == COOP_COLUMNS:
        return str(COOP_VECTOR_LENGTH)
    return COOP_ENUMERATORS.get(kind.opaque, "")


def coop_element(opaque: str) -> Type | None:
    """The element type of the cooperative vector the opaque spelling ``opaque`` names, or ``None``."""
    match = _COOP_VECTOR.fullmatch(opaque)
    return Type(match.group(1)) if match else None


_TUPLE = re.compile(r"Tuple<(\w+)((?:, \1)*)>")


def tuple_of(scalar: str, length: int) -> Type:
    """Slang's tuple of ``length`` ``scalar`` elements: an opaque type that keeps its element type as ``scalar``, so a
    probe can fill one from the input buffers."""
    return Type(scalar, (), "Tuple<" + ", ".join([scalar] * length) + ">")


def tuple_element(opaque: str) -> tuple[Type, int] | None:
    """The element type and length of the tuple the opaque spelling ``opaque`` names, or ``None``."""
    match = _TUPLE.fullmatch(opaque)
    if match is None or match.group(1) not in SCALARS:
        return None
    return Type(match.group(1)), match.group(2).count(",") + 1


@dataclass(frozen=True)
class Param:
    """A parameter; ``direction`` is ``in``, ``out``, ``inout`` (an atomic or read-modify-write operand),
    ``const``, an argument the language requires to be a constant expression, ``address``, a pointer to a local
    value that the function reads and may write, ``pointer``, a pointer to device memory, which the probe declares as
    a uniform, ``workgroup``, a variable the compute workgroup shares, or ``buffer``, a runtime-sized array, for which
    the probe passes its input buffer of the parameter's type.
    """

    type: Type
    direction: str = "in"


@dataclass(frozen=True)
class Signature:
    result: Type
    params: tuple[Param, ...]

    @property
    def unsupported(self) -> str:
        """The first type no probe can express yet, or an empty string."""
        for kind in (self.result, *(p.type for p in self.params)):
            if kind.opaque and kind != VOID:
                return kind.opaque
        return ""


@dataclass(frozen=True)
class Feature:
    """One entry of a target language's feature list.

    ``name`` is the upstream spelling and ``group`` the upstream section it comes from. ``signature`` is the overload
    the probe uses and ``crossgl`` the CrossGL spelling of the same operation, empty when CrossGL has none; ``note``
    then says why. ``stage`` is the shader stage the probe must run in. ``spelling`` is how the native probe calls
    the feature when that differs from ``name``, as for WGSL's ``bitcast<u32>``. ``counterpart`` is the overload the
    CrossGL probe calls when CrossGL's operation takes other arguments than the feature, as a WGSL texture function
    does: it takes a texture and a sampler where CrossGL takes one GLSL sampler. ``target`` is the compiler target
    probes are checked for when it is not the language's usual one, as ``hlsl`` for a Slang function that SPIR-V
    lacks or ``vulkan`` for a GLSL function that only GLSL for Vulkan has.
    """

    language: str
    category: str
    name: str
    signature: Signature | None
    group: str = ""
    crossgl: str = ""
    stage: str = "compute"
    note: str = ""
    spelling: str = ""
    counterpart: Signature | None = None
    target: str = ""

    @property
    def key(self) -> str:
        return f"{self.language}/{self.category}/{self.name}"


@dataclass
class Result:
    feature: Feature
    direction: str
    outcome: str
    detail: str = ""
    validator: str = ""
    # The semantic feature id (:mod:`.semantic`), filled in for the output direction; empty when there is none.
    semantic: str = ""
    # The native probe and the translation the validator accepted, which :mod:`.execution` runs; empty otherwise.
    sources: tuple[str, str] = field(default=(), compare=False, repr=False)
    # The runtime :mod:`.execution` ran both on, when it did.
    runtime: str = ""

    def __post_init__(self):
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}")


@dataclass
class Score:
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(OUTCOMES, 0))

    def add(self, outcome: str) -> None:
        self.counts[outcome] += 1

    @property
    def measured(self) -> int:
        return self.counts["supported"] + self.counts["refused"] + self.counts["wrong"]

    @property
    def percent(self) -> float | None:
        """``supported / (supported + refused + wrong)``; ``None`` when nothing was measured."""
        if not self.measured:
            return None
        return 100.0 * self.counts["supported"] / self.measured

"""Value types shared by the feature lists, probes and report."""

from __future__ import annotations

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
    "int8x4_packed",
    "uint8x4_packed",
)
# Scalars a probe reads from and stores as 32-bit buffer elements: the 64-bit integers, the 16-bit scalars, and the
# 32-bit words that pack four 8-bit integers (HLSL shader model 6.6).
WIDE_INTEGERS = frozenset({"int64", "uint64"})
NARROW_SCALARS = frozenset({"half", "int16", "uint16"})
PACKED_BYTES = frozenset({"int8x4_packed", "uint8x4_packed"})
# Overload selection prefers a signature without these when a function has one.
OTHER_WIDTHS = frozenset({"double"}) | WIDE_INTEGERS | NARROW_SCALARS
OUTCOMES = ("supported", "refused", "wrong", "untested")
LANGUAGES = (
    "cuda",
    "glsl",
    "hip",
    "hlsl",
    "metal",
    "mojo",
    "rust",
    "slang",
    "spirv",
    "wgsl",
)
DIRECTIONS = ("output", "input")
# The ray-tracing stages a probe can run in, each with the name HLSL's ``[shader(...)]`` attribute and slangc's
# ``-stage`` give it. A ray-tracing intrinsic is probed in a closest-hit shader, which allows most of them, unless only
# one other stage has it (DirectX Raytracing functional specification, "Intrinsics", and HLSL shader execution
# reordering).
RAYTRACING_STAGES = {
    "ray-generation": "raygeneration",
    "intersection": "intersection",
    "any-hit": "anyhit",
    "closest-hit": "closesthit",
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
    """A value type: a scalar, a vector ``(n,)`` or a matrix ``(columns, rows)``.

    Any other kind of type, such as a sampler or an image, is carried as ``opaque`` with its upstream spelling. A
    probe can pass only the samplers, images and objects that :mod:`.objects` declares.
    """

    scalar: str
    shape: tuple[int, ...] = ()
    opaque: str = ""

    def __post_init__(self):
        if not self.opaque and self.scalar not in SCALARS:
            raise ValueError(f"unknown scalar type {self.scalar!r}")
        if len(self.shape) > 2 or any(not 2 <= n <= 4 for n in self.shape):
            raise ValueError(f"unsupported shape {self.shape!r}")

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


@dataclass(frozen=True)
class Param:
    """A parameter; ``direction`` is ``in``, ``out``, ``inout`` (an atomic or read-modify-write operand) or
    ``const``, an argument the language requires to be a constant expression."""

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
    does: it takes a texture and a sampler where CrossGL takes one GLSL sampler.
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

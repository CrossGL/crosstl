"""The resources a probe declares: object types whose methods it calls, such as HLSL's ``Texture2D`` with
``SampleLevel``, and the GLSL samplers and images a built-in function takes.

A method probe declares one object, named ``obj``, and the samplers its overload takes, then calls the method once on
arguments read from the input buffers, as a function probe calls a function. Resources are bound to registers past
the probe's buffers; a ray query is a local variable. The language's dialect spells the declaration
(:meth:`~tools.coverage.dialect.Dialect.resource`); CrossGL declares the same object types and calls the same methods.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dialect import Dialect, ProbeError
from .model import Type


@dataclass(frozen=True)
class Object:
    """``spelling`` names the type, with ``{element}`` for its element type; ``register`` is the register class it
    binds to, empty for an object that is a local variable."""

    spelling: str
    element: Type | None = None
    register: str = ""


_FLOAT4 = Type("float", (4,))


def _textures(spelling: str, register: str, *names: str) -> dict[str, Object]:
    return {
        name: Object(spelling.format(name=name), _FLOAT4, register) for name in names
    }


# Keyed by the upstream type name. DXC's table writes ``TextureCUBE`` for the type HLSL spells ``TextureCube``, and
# ``Texture2DArrayMS`` for ``Texture2DMSArray``.
OBJECTS: dict[str, Object] = {
    **_textures(
        "{name}<{{element}}>",
        "t",
        "Texture1D",
        "Texture1DArray",
        "Texture2D",
        "Texture2DArray",
        "Texture3D",
        "Texture2DMS",
        "Buffer",
    ),
    "Texture2DArrayMS": Object("Texture2DMSArray<{element}>", _FLOAT4, "t"),
    "TextureCUBE": Object("TextureCube<{element}>", _FLOAT4, "t"),
    "TextureCUBEArray": Object("TextureCubeArray<{element}>", _FLOAT4, "t"),
    **_textures(
        "{name}<{{element}}>",
        "u",
        "RWTexture1D",
        "RWTexture1DArray",
        "RWTexture2D",
        "RWTexture2DArray",
        "RWTexture3D",
        "RWTexture2DMS",
        "RWTexture2DMSArray",
        "RWBuffer",
    ),
    "StructuredBuffer": Object("StructuredBuffer<{element}>", Type("float"), "t"),
    "RWStructuredBuffer": Object("RWStructuredBuffer<{element}>", Type("float"), "u"),
    "AppendStructuredBuffer": Object(
        "AppendStructuredBuffer<{element}>", Type("float"), "u"
    ),
    "ConsumeStructuredBuffer": Object(
        "ConsumeStructuredBuffer<{element}>", Type("float"), "u"
    ),
    "ByteAddressBuffer": Object("ByteAddressBuffer", None, "t"),
    "RWByteAddressBuffer": Object("RWByteAddressBuffer", None, "u"),
    "RayQuery": Object("RayQuery<RAY_FLAG_NONE>"),
}
# Sampler parameters, by the opaque type a feature list gives them: the sampler type, its name and its register.
SAMPLERS = {
    "sampler": ("SamplerState", "smp", "s0"),
    "sampler_cmp": ("SamplerComparisonState", "cmp", "s1"),
}
# The probe's buffers take the first registers of their class; resources bind after them.
_REGISTERS = {"t": "t0", "u": "u8"}


# GLSL's opaque uniform types that a function probe can pass, with the image format each is declared with. Probes
# use the 2D forms; the others stay untested.
UNIFORMS = {
    "sampler2D": "",
    "sampler2DMS": "",
    "image2D": "rgba32f",
    "image2DMS": "rgba32f",
    "iimage2D": "r32i",
    "uimage2D": "r32ui",
}
# Uniforms bind past the probe's buffers.
_FIRST_UNIFORM = 8


class Uniforms:
    """The samplers and images a function probe passes, declared the first time each parameter needs one."""

    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.declarations: list[str] = []

    @staticmethod
    def takes(kind: Type) -> bool:
        return kind.opaque in UNIFORMS

    def argument(self, kind: Type) -> str:
        name = f"u{len(self.declarations)}"
        binding = _FIRST_UNIFORM + len(self.declarations)
        self.declarations.append(
            self.dialect.uniform(kind.opaque, name, binding, UNIFORMS[kind.opaque])
        )
        return name


class Receiver:
    """The object a method probe calls its method on, and the samplers the call takes."""

    name = "obj"

    def __init__(self, owner: str, dialect: Dialect):
        kind = OBJECTS.get(owner)
        if kind is None:
            raise ProbeError(f"no probe for {owner} objects yet")
        self.dialect = dialect
        element = dialect.type(kind.element) if kind.element else ""
        spelling = kind.spelling.format(element=element)
        self.declarations: list[str] = []
        self.locals: list[str] = []
        if kind.register:
            self.declarations.append(
                dialect.resource(spelling, self.name, _REGISTERS[kind.register])
            )
        else:
            self.locals.append(f"{spelling} {self.name};")
        self._samplers: set[str] = set()

    @staticmethod
    def element(owner: str) -> Type | None:
        kind = OBJECTS.get(owner)
        return kind.element if kind else None

    def takes(self, kind: Type) -> bool:
        return kind.opaque in SAMPLERS

    def argument(self, kind: Type) -> str:
        """The sampler to pass for a parameter of ``kind``, declared the first time it is used."""
        spelling, name, register = SAMPLERS[kind.opaque]
        if name not in self._samplers:
            self._samplers.add(name)
            self.declarations.append(self.dialect.resource(spelling, name, register))
        return name

    def call(self, method: str) -> str:
        return f"{self.name}.{method}"

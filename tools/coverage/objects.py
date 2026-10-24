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
    binds to, empty for an object that is a local variable; ``stage`` is the stage its methods are probed in.
    ``parameter`` marks an object the stage's entry point takes, which the dialect's stage program declares.
    """

    spelling: str
    element: Type | None = None
    register: str = ""
    stage: str = "compute"
    parameter: bool = False


_FLOAT4 = Type("float", (4,))
# The vertex type a geometry probe's output stream holds. DXC's table writes the type of the vertex ``Append`` takes
# as the stream's own element type; a probe appends the vertex its geometry program fills in.
VERTEX = "Vertex"
STREAM = f"PointStream<{VERTEX}>"
# The record type of a work-graph probe's node inputs and outputs.
RECORD = "Record"
# The hit object of shader execution reordering, as DXC's table names it.
HIT_OBJECT = "DxHitObject"


def _node(spelling: str) -> Object:
    return Object(spelling, stage="node", parameter=True)


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
    # A default-constructed hit object is a NOP hit object (HLSL shader execution reordering), which a ray-generation
    # shader can query.
    HIT_OBJECT: Object("dx::HitObject", stage="ray-generation"),
    # A geometry shader's output stream is a parameter of its entry point.
    "Stream": Object(STREAM, stage="geometry", parameter=True),
    # A work graph's node shader takes its inputs and outputs as entry-point parameters (HLSL work graphs). The
    # records object OutputComplete is called on is one the node requests from an output.
    "NodeOutput": _node(f"NodeOutput<{RECORD}>"),
    "EmptyNodeOutput": _node("EmptyNodeOutput"),
    "EmptyNodeInput": _node("EmptyNodeInput"),
    "GroupNodeInputRecords": _node(f"GroupNodeInputRecords<{RECORD}>"),
    "RWDispatchNodeInputRecord": _node(f"RWDispatchNodeInputRecord<{RECORD}>"),
    "GroupOrThreadNodeOutputRecords": _node(f"ThreadNodeOutputRecords<{RECORD}>"),
    # A feedback texture records which mips of its paired texture a sample would touch; it binds as a UAV.
    "FeedbackTexture2D": Object(
        "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP>", None, "u"
    ),
    "FeedbackTexture2DArray": Object(
        "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIN_MIP>", None, "u"
    ),
}
# The samplers a method takes, and the texture a feedback texture's methods sample, by the opaque type a feature
# list gives them: the declared type, its name and its register.
SAMPLERS = {
    "sampler": ("SamplerState", "smp", "s0"),
    "sampler_cmp": ("SamplerComparisonState", "cmp", "s1"),
    "Texture2D": ("Texture2D<float4>", "tex", "t0"),
    "Texture2DArray": ("Texture2DArray<float4>", "tex", "t0"),
}
# The probe's buffers take the first registers of their class; resources bind after them.
_REGISTERS = {"t": "t0", "u": "u8"}


@dataclass(frozen=True)
class Uniform:
    """A sampler, texture, image, subpass input or atomic counter a function probe declares at module scope;
    ``qualifier`` is the GLSL layout qualifier it needs besides its binding (an image's format, a subpass input's
    attachment index), and ``binding`` a binding the uniform must use.
    """

    spelling: str
    qualifier: str = ""
    binding: int | None = None


def _same(*names: str) -> dict[str, Uniform]:
    return {name: Uniform(name) for name in names}


_GLSL_UNIFORMS = {
    **_same("sampler2D", "sampler2DMS"),
    "image2D": Uniform("image2D", "rgba32f"),
    "image2DMS": Uniform("image2DMS", "rgba32f"),
    "iimage2D": Uniform("iimage2D", "r32i"),
    "uimage2D": Uniform("uimage2D", "r32ui"),
    "subpassInput": Uniform("subpassInput", "input_attachment_index = 0"),
    # Atomic counters bind in a namespace of their own, and glslang's default limits allow only binding 0
    # (gl_MaxAtomicCounterBindings).
    "atomic_uint": Uniform("atomic_uint", binding=0),
}
# Tint's table writes a texture's texel type as ``T`` where any sampled type will do; a probe samples ``f32``.
_WGSL_UNIFORMS = {
    **_same("sampler", "sampler_comparison", "texture_depth_2d", "texture_2d<f32>"),
    "texture_2d<T>": Uniform("texture_2d<f32>"),
    "texture_2d_array<T>": Uniform("texture_2d_array<f32>"),
    "texture_2d_array<f32>": Uniform("texture_2d_array<f32>"),
    "texture_multisampled_2d<T>": Uniform("texture_multisampled_2d<f32>"),
    "texture_storage_2d<F, W>": Uniform("texture_storage_2d<rgba32float, write>"),
}


def _acceleration_structure(spelling: str) -> dict[str, Uniform]:
    """A ray-tracing acceleration structure of type ``spelling``, under the names the HLSL and Slang lists give it."""
    return {
        name: Uniform(spelling)
        for name in ("acceleration_struct", "RaytracingAccelerationStructure")
    }


# The opaque parameter types a function probe can pass, by language and as each feature list spells them. Probes use
# the 2D forms; functions that only take the others stay untested. CrossGL calls texture functions as GLSL does.
UNIFORMS: dict[str, dict[str, Uniform]] = {
    "glsl": _GLSL_UNIFORMS,
    # The CrossGL counterparts of WGSL's depth and array textures (:data:`.spellings.COUNTERPARTS`) take two more.
    # CrossGL spells an acceleration structure as every generator reads it, and byte-address buffers as Slang does.
    "crossgl": {
        **_GLSL_UNIFORMS,
        **_same("sampler2DShadow", "sampler2DArray"),
        **_acceleration_structure("accelerationStructureEXT"),
        **_same("ByteAddressBuffer", "RWByteAddressBuffer"),
    },
    "hlsl": _acceleration_structure("RaytracingAccelerationStructure"),
    # Slang's cooperative-vector functions load and accumulate through byte-address buffers.
    "slang": {
        **_acceleration_structure("RaytracingAccelerationStructure"),
        **_same("ByteAddressBuffer", "RWByteAddressBuffer"),
    },
    "wgsl": _WGSL_UNIFORMS,
    # OpenCL C passes an image as a kernel argument and declares a sampler as a program-scope constant.
    "opencl": _same(
        "read_only image2d_t",
        "write_only image2d_t",
        "read_write image2d_t",
        "sampler_t",
    ),
    # GLSL ES 3.00 has no images or multisampled samplers; the probe passes the samplers CrossGL also declares.
    "webgl": _same("sampler2D", "sampler2DShadow", "sampler2DArray"),
}
# Uniforms bind past the probe's buffers.
_FIRST_UNIFORM = 8


class Uniforms:
    """The samplers, textures and images a function probe passes, declared the first time each parameter needs
    one."""

    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.table = UNIFORMS.get(dialect.language, {})
        self.declarations: list[str] = []

    def takes(self, kind: Type) -> bool:
        return kind.opaque in self.table

    def argument(self, kind: Type) -> str:
        uniform = self.table[kind.opaque]
        name = f"u{len(self.declarations)}"
        binding = (
            _FIRST_UNIFORM + len(self.declarations)
            if uniform.binding is None
            else uniform.binding
        )
        self.declarations.append(
            self.dialect.uniform(uniform.spelling, name, binding, uniform.qualifier)
        )
        return name


# Methods whose object must hold another element type: GatherRaw returns raw texel bits, so its texture holds uint.
_ELEMENTS = {"GatherRaw": Type("uint")}


def element(owner: str, method: str) -> Type | None:
    """The element type of the object ``method`` is probed on."""
    kind = OBJECTS.get(owner)
    if kind is None or kind.element is None:
        return None
    return _ELEMENTS.get(method, kind.element)


# What a probe does with a local object it passes, so that a live instruction uses it. DXC's validator counts a ray
# query in the shader's flags even when the optimizer removes every use, then rejects the shader ("Flags must match
# usage"), so a ray query is advanced once.
_STARTS = {"RayQuery": "{name}.Proceed();"}


def local_object(kind: Type, name: str = "obj") -> list[str]:
    """The statements that declare ``name`` as the default-constructed local object a parameter of ``kind`` takes,
    such as a ray query or a hit object; none when ``kind`` is no such object."""
    found = OBJECTS.get(kind.opaque)
    if found is None or found.register or found.parameter or found.element:
        return []
    start = _STARTS.get(kind.opaque)
    return [f"{found.spelling} {name};", *([start.format(name=name)] if start else [])]


class Receiver:
    """The object a method probe calls its method on, and the samplers the call takes."""

    name = "obj"

    def __init__(self, owner: str, dialect: Dialect, method: str = ""):
        kind = OBJECTS.get(owner)
        if kind is None:
            raise ProbeError(f"no probe for {owner} objects yet")
        self.dialect = dialect
        held = element(owner, method)
        spelling = kind.spelling.format(element=dialect.type(held) if held else "")
        self.declarations: list[str] = []
        self.locals: list[str] = []
        # The type of the entry-point parameter the stage program declares as the object, if it is one.
        self.parameter = spelling if kind.parameter else ""
        if kind.register:
            self.declarations.append(
                dialect.resource(spelling, self.name, _REGISTERS[kind.register])
            )
        elif not kind.parameter:
            self.locals.append(f"{spelling} {self.name};")
        self._samplers: set[str] = set()

    def takes(self, kind: Type) -> bool:
        return kind.opaque in SAMPLERS or bool(self.parameter and kind.opaque == VERTEX)

    def argument(self, kind: Type) -> str:
        """The sampler or texture to pass for a parameter of ``kind``, declared the first time it is used, or the
        vertex the geometry program fills in."""
        if kind.opaque == VERTEX:
            return self.dialect.vertex
        spelling, name, register = SAMPLERS[kind.opaque]
        if name not in self._samplers:
            self._samplers.add(name)
            self.declarations.append(self.dialect.resource(spelling, name, register))
        return name

    def call(self, method: str) -> str:
        """The callee of ``method``: the object's own method, or a static method its type spells."""
        return method if "::" in method else f"{self.name}.{method}"

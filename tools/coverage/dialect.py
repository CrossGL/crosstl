"""How one language spells a probe: types, expressions, statements and the kernel around them.

:class:`Dialect` spells everything the way GLSL does; a language overrides only what it writes differently. The probe
builder in :mod:`.probes` talks to a dialect and never to a language directly.
"""

from __future__ import annotations

from typing import NamedTuple

from .model import (
    BYTES,
    NARROW_SCALARS,
    PACKED_BYTES,
    POINTER_SIZED,
    STRING,
    WIDE_INTEGERS,
    Type,
)

# A storage buffer as ``(element type spelling, name)``; buffer ``k`` is bound at binding ``k``.
Buffer = tuple[str, str]


class Handle(NamedTuple):
    """An argument the language makes itself, such as an event or a queue that another built-in returns: the
    argument, the statements before the call that make it, and the declarations before the entry point it needs.
    """

    argument: str
    lines: tuple[str, ...] = ()
    declarations: tuple[str, ...] = ()


class ProbeError(ValueError):
    """The feature's signature cannot be expressed as a probe yet."""


class NoSpelling(ProbeError):
    """The language has no spelling for the operation, so the operation is a gap rather than an untested probe."""


# GLSL ES 3.00 has neither storage buffers nor a compute stage (GLSL ES 3.00.6, sections 2 and 4.3), so a WebGL probe,
# in GLSL ES and in CrossGL, is a fragment shader that reads its inputs from uniform arrays of this length and writes
# its result to a local array.
UNIFORM_ARRAY_LENGTH = 32


def uniform_read(buffer: str, offset: int) -> str:
    """Element ``offset`` of the uniform array that stands in for the input buffer ``buffer``."""
    if offset >= UNIFORM_ARRAY_LENGTH:
        raise ProbeError(
            f"the probe reads more than {UNIFORM_ARRAY_LENGTH} elements of {buffer}"
        )
    return f"{buffer}[i + {offset}u]"


class Dialect:
    language = ""
    suffix = ""
    entry = "main"
    # The file suffix of a fragment or geometry probe, where the language's tools tell the stage from it.
    fragment_suffix = ""
    geometry_suffix = ""
    # The task payload a task-stage probe declares and hands to the mesh workgroups it launches.
    payload = "payload"
    # The vertex a geometry probe fills in from its input point, for a stream's ``Append``.
    vertex = "emitted"
    # HLSL builds matrices from rows and indexes rows; the others use columns.
    rows_first = False
    _PREFIX = {"float": "", "double": "d", "int": "i", "uint": "u", "bool": "b"}
    # Whether the dialect spells the scalars in WIDE_INTEGERS, NARROW_SCALARS, BYTES, PACKED_BYTES and POINTER_SIZED.
    wide_integers = False
    narrow_scalars = False
    byte_scalars = False
    packed_bytes = False
    pointer_sized = False
    # The spelling of a constant string argument (model.STRING), empty where the language has no strings.
    string = ""
    # How the dialect spells an unsigned one, for ``constant``.
    one = "1u"

    def type(self, kind: Type) -> str:
        if kind.count:
            raise ProbeError(f"no probe for {self.language} array parameters yet")
        if kind == STRING and self.string:
            return self.string
        if kind.opaque:
            if kind.opaque == "void":
                return "void"
            raise ProbeError(f"no probe for parameters of type {kind.opaque}")
        if kind.scalar in WIDE_INTEGERS and not self.wide_integers:
            raise ProbeError(f"no 64-bit integer probe for {self.language} yet")
        if kind.scalar in NARROW_SCALARS and not self.narrow_scalars:
            raise ProbeError(f"no 16-bit probe for {self.language} yet")
        if kind.scalar in BYTES and not self.byte_scalars:
            raise ProbeError(f"no 8-bit probe for {self.language} yet")
        if kind.scalar in PACKED_BYTES and not self.packed_bytes:
            raise ProbeError(f"{self.language} has no packed 8-bit integer types")
        if kind.scalar in POINTER_SIZED and not self.pointer_sized:
            raise ProbeError(f"{self.language} has no pointer-sized integer type")
        if not kind.shape:
            return self.scalar(kind.scalar)
        if len(kind.shape) == 1:
            return self.vector(kind.scalar, kind.shape[0])
        return self.matrix(kind.scalar, *kind.shape)

    def scalar(self, scalar: str) -> str:
        return scalar

    def vector(self, scalar: str, size: int) -> str:
        return f"{self._PREFIX[scalar]}vec{size}"

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        if scalar not in ("float", "double"):
            raise ProbeError(f"{self.language} has no {scalar} matrices")
        shape = f"{columns}" if columns == rows else f"{columns}x{rows}"
        return f"{self._PREFIX[scalar]}mat{shape}"

    def read(self, buffer: str, offset: int) -> str:
        return f"{buffer}[i + {offset}u]"

    def positive(self, value: str) -> str:
        return f"({value} > 0.0)"

    def construct(self, kind: Type, parts: list[str]) -> str:
        return f"{self.type(kind)}({', '.join(parts)})"

    def cast(self, kind: Type, value: str) -> str:
        return f"{self.type(kind)}({value})"

    def array(self, kind: Type, parts: list[str]) -> str:
        """An array of the type ``kind`` holding ``parts``, for a parameter that takes one."""
        raise ProbeError(f"no probe for {self.language} array parameters yet")

    def literal(self, kind: Type) -> str:
        """A constant of ``kind``, for an argument the language requires to be a constant expression."""
        if kind == STRING:
            return '""'
        if len(kind.shape) > 1:
            raise ProbeError("no constant probe argument of a matrix type")
        constants = {
            "bool": "false",
            "int": "0",
            "int16": "0",
            "int64": "0",
            "uint": "0u",
            "uint16": "0u",
            "uint64": "0u",
            "int8": "0",
            "uint8": "0u",
            "int8x4_packed": "0u",
            "uint8x4_packed": "0u",
        }
        zero = constants.get(kind.scalar, "0.0")
        if kind.shape:
            return self.construct(kind, [zero] * kind.shape[0])
        return zero

    def constant(self, kind: Type) -> str:
        """An argument of ``kind`` that the language requires to be a constant expression. An unsigned scalar is a
        count, a size or an index, and one is the smallest value all of them take: a cluster size must be a power of
        two (SPIR-V 1.6, OpGroupNonUniformRotateKHR)."""
        if kind.scalar in ("uint", "uint16", "uint64", "uint8") and not kind.shape:
            return self.one
        return self.literal(kind)

    def constant_array(self, kind: Type, name: str) -> str:
        """A local constant ``name`` of the array type ``kind``, for an argument that must be a constant
        expression."""
        element = self.type(kind.element)
        values = ", ".join([self.literal(kind.element)] * kind.count)
        return (
            f"const {element} {name}[{kind.count}] = {element}[{kind.count}]({values});"
        )

    def buffer_element(self, kind: Type, atomic: bool) -> str:
        """The element type of the output buffer; ``atomic`` when the probe passes it to an atomic operation."""
        return self.type(kind)

    def absent(self, stage: str) -> str:
        """Why the language has no ``stage`` at all, or an empty string when it has one or a probe is missing."""
        return ""

    def holding(self, opaque: str) -> str:
        """The spelling of a local that holds a result of the opaque type ``opaque``, for a call whose result must be
        kept, or an empty string when the probe only makes the call."""
        return ""

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        """How a probe reads a result of the opaque type ``opaque`` as a value: the value's type and an expression
        with ``{}`` for the call, or ``None`` when the probe only makes the call."""
        return None

    def stored(self, kind: Type, value: str) -> str:
        """``value``, of type ``kind``, as the output buffer's element type holds it."""
        return value

    def keep(self, atomic: bool) -> str:
        """The baseline probe's store: it keeps the output buffer in use without computing anything."""
        return self.store("outp[i]")

    def declare(self, kind: Type, name: str, value: str = "") -> str:
        initial = f" = {value}" if value else ""
        return f"{self.type(kind)} {name}{initial};"

    def resource(self, spelling: str, name: str, register: str) -> str:
        """The declaration of a resource object, such as a texture or a sampler, of type ``spelling``."""
        raise ProbeError(f"no probe for {self.language} {spelling} objects yet")

    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        """The declaration of a GLSL sampler, image or subpass input uniform; ``qualifier`` is the layout qualifier
        it needs besides its binding."""
        raise ProbeError(f"no probe for {self.language} {spelling} uniforms yet")

    def call(self, name: str, arguments: list[str]) -> str:
        return f"{name}({', '.join(arguments)})"

    def assignment(
        self, name: str, kind: Type, arguments: list[str]
    ) -> tuple[list[str], str] | None:
        """For a function this dialect writes as an assignment rather than a call, the statements that compute its
        result of type ``kind``, and the local that holds it; None for a call."""
        return None

    def out_argument(self, name: str) -> str:
        return name

    def address(self, name: str) -> str:
        """A pointer to the local ``name``, for a parameter that takes one."""
        raise ProbeError(f"no probe passes {self.language} a pointer yet")

    def pointer(self, kind: Type, name: str) -> str:
        """The declaration of a uniform ``name`` that holds a pointer to device memory of ``kind``, for a parameter
        that takes one."""
        raise ProbeError(f"no probe passes {self.language} a device pointer yet")

    def workgroup(self, kind: Type, name: str) -> str:
        """The declaration of a variable ``name`` that a compute workgroup shares."""
        count = f"[{kind.count}]" if kind.count else ""
        return f"shared {self.type(kind.element)} {name}{count};"

    def workgroup_argument(self, name: str) -> str:
        """The workgroup variable ``name``, for a parameter that takes one."""
        return name

    def handle(self, kind: Type, name: str) -> Handle | None:
        """How a probe makes an argument of the opaque type ``kind`` in a local ``name``, or ``None`` when the
        language has no built-in that makes one."""
        return None

    def inout_argument(self) -> str:
        return "outp[i]"

    def buffer_argument(self, name: str) -> str:
        """The whole input buffer ``name``, for a parameter that takes a runtime-sized array."""
        return name

    def to_int(self, condition: str) -> str:
        return f"{condition} ? 1 : 0"

    def first(self, vector: str) -> str:
        return f"{vector}.x"

    def part(self, matrix: str) -> str:
        """The first column (or, for :attr:`rows_first` languages, row) of a matrix."""
        return f"{matrix}[0]"

    def store(self, value: str) -> str:
        return f"outp[i] = {value};"

    def statement(self, expression: str) -> str:
        return f"{expression};"

    def flat(self, kind: str) -> str:
        """A shader input of type ``kind`` that is not interpolated."""
        return f"flat {kind}"

    def stage_input(self, name: str) -> str:
        """How a fragment shader reads its input ``name``, which :meth:`fragment` declares from ``inputs``."""
        return name

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        """A compute kernel that runs ``body`` once per invocation, with ``i`` the invocation's index.

        ``parts`` may add ``prelude`` lines (extensions, enable directives, or for CrossGL shader-level declarations),
        ``declarations`` that precede the entry point and, for languages that pass them as entry-point parameters,
        ``params``.
        """
        raise NotImplementedError

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        """A fragment shader that runs ``body`` with ``i`` set to 0 and writes a constant colour.

        Fragment-only features, such as derivatives, are probed here; the buffers are the same as :meth:`kernel`'s.
        ``parts`` may add ``inputs``, the shader inputs as ``(type spelling, name)``, for a function that takes one.
        """
        raise ProbeError(f"{self.language} has no fragment stage")

    def geometry(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        """A geometry shader that takes and emits points and runs ``body`` with ``i`` set to 0, for a feature only
        that stage has; the buffers and ``parts`` are :meth:`kernel`'s. ``parts`` may add ``parameter``, the type of
        the output stream the entry point takes as ``obj``, for a probe of the stream's methods; the probe then fills
        in :attr:`vertex` before ``body``."""
        raise ProbeError(f"no geometry-stage probe for {self.language} yet")

    def raytracing(
        self, stage: str, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        """A shader of the ray-tracing ``stage`` that runs ``body`` with ``i`` set to 0, for a feature only that stage
        has; the buffers and ``parts`` are :meth:`kernel`'s."""
        raise ProbeError(f"no {stage}-stage probe for {self.language} yet")

    def tessellation_control(
        self, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        """A tessellation-control shader for a triangle patch whose patch-constant code runs ``body`` with ``i`` set
        to 0; the buffers and ``parts`` are :meth:`kernel`'s."""
        raise ProbeError(f"no tessellation-control-stage probe for {self.language} yet")

    def mesh(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        """A mesh shader that runs ``body`` with ``i`` set to 0 and then writes one triangle; the buffers and ``parts``
        are :meth:`kernel`'s."""
        raise ProbeError(f"no mesh-stage probe for {self.language} yet")

    def task(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        """A task shader that runs ``body`` with ``i`` set to 0. It declares the task payload :attr:`payload`, which
        ``body`` hands to the mesh workgroups it launches; the buffers and ``parts`` are :meth:`kernel`'s.
        """
        raise ProbeError(f"no task-stage probe for {self.language} yet")

    def node(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        """A work-graph node shader of one thread, launched once, that runs ``body`` with ``i`` set to 0; the
        buffers and ``parts`` are :meth:`kernel`'s. ``parts`` may add ``parameter``, the type of the node input or
        output the entry point takes as ``obj``, for a probe of its methods."""
        raise ProbeError(f"no node-stage probe for {self.language} yet")

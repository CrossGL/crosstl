"""Minimal probe programs: one call of one feature inside a compute kernel, or a shader of the one stage that has the
feature.

Arguments are read from input buffers, so no call folds to a constant, and the result is written to an output
buffer, so no call is dead code. Output probes are CrossGL (:mod:`.crossgl`); input probes are written in the source
language. A *baseline* probe reads the same arguments but makes no call; :mod:`.semantic` subtracts it to find the
instructions the call itself lowers to.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import constructs
from .dialect import Buffer, Dialect, NoSpelling, ProbeError
from .languages import DIALECTS
from .model import RAYTRACING_STAGES, Feature, Param, Signature, Type, tuple_element
from .objects import Receiver, Uniforms, local_object

__all__ = ["DIALECTS", "NoSpelling", "Probe", "ProbeError", "build", "spell"]
_BUFFERS = {"float": "fin", "int": "iin", "uint": "uin"}
# Fragment-shader functions whose first argument must be a shader input itself, not a value computed from one.
_INTERPOLANTS = frozenset(
    "interpolateAtCentroid interpolateAtOffset interpolateAtSample InterpolateAtCentroid InterpolateAtOffset "
    "InterpolateAtSample EvaluateAttributeAtCentroid EvaluateAttributeAtSample EvaluateAttributeCentroid "
    "EvaluateAttributeSnapped GetAttributeAtVertex".split()
)
# Those whose input must not be interpolated, because the function reads one vertex's value.
_FLAT = frozenset({"GetAttributeAtVertex"})
# Stages whose probe is one dialect method, which takes the buffers, the body and the declarations.
_STAGE_PROGRAMS = {
    "geometry": "geometry",
    "tessellation-control": "tessellation_control",
    "mesh": "mesh",
    "task": "task",
    "node": "node",
}
# Stages whose probe declares a payload, which a parameter of a user-defined or by-reference type takes: the task
# payload a task shader hands to its mesh workgroups, the ray payload a hit shader receives and can trace with, and
# the data a callable shader receives and can pass to another.
_PAYLOAD_STAGES = frozenset({"task", "any-hit", "closest-hit", "callable"})
# The stage that starts rays: its probe passes a user-defined inout parameter a ray payload it declares as a local.
_TRACING_STAGE = "ray-generation"
# A ray descriptor, as the HLSL and Slang feature lists spell its type. A probe fills one in field by field.
_RAY_DESCRIPTIONS = frozenset({"ray_desc", "RayDesc"})
# The hit attributes a probe passes for a user-defined type outside the payload stages: those ``ReportHit`` reports
# from an intersection shader and a hit object hands back. HLSL, Slang and CrossGL spell the triangle attributes alike.
_HIT_ATTRIBUTES = "BuiltInTriangleIntersectionAttributes"


@dataclass(frozen=True)
class Probe:
    language: str
    source: str
    entry: str
    suffix: str


def spell(kind: Type, language: str) -> str:
    return DIALECTS[language].type(kind)


# Scalars no input or output buffer holds: they are read from, and stored as, a 32-bit buffer element.
_STORAGE = {
    "bool": "float",
    "double": "float",
    "half": "float",
    "int8": "int",
    "int16": "int",
    "int64": "int",
    "uint8": "uint",
    "uint16": "uint",
    "uint64": "uint",
    "int8x4_packed": "uint",
    "uint8x4_packed": "uint",
    "usize": "uint",
}


def _storage(scalar: str) -> str:
    return _STORAGE.get(scalar, scalar)


class _Builder:
    def __init__(
        self,
        dialect: Dialect,
        receiver: Receiver | None = None,
        interpolant: str = "",
        payload: bool = False,
        tracing: bool = False,
    ):
        self.dialect = dialect
        self.receiver = receiver
        self.interpolant = interpolant
        # Whether the probe declares a payload (:data:`_PAYLOAD_STAGES`).
        self.payload = payload
        # Whether the probe can declare a payload local (:data:`_TRACING_STAGE`), and whether a call passes one.
        self.tracing = tracing
        self.traced = False
        self.uniforms = Uniforms(dialect)
        self.offsets: dict[str, int] = {}
        self.lines: list[str] = list(receiver.locals) if receiver else []
        # Variables the compute workgroup shares, for a parameter that takes one.
        self.shared: list[str] = []
        # Uniforms that hold pointers to device memory, for a parameter that takes one.
        self.pointers: list[str] = []
        # What the arguments the dialect makes itself declare before the entry point (Dialect.handle).
        self.handles: list[str] = []
        # Fragment-shader inputs as ``(type spelling, name)``.
        self.inputs: list[Buffer] = []

    def element(self, scalar: str) -> str:
        storage = _storage(scalar)
        offset = self.offsets.get(storage, 0)
        self.offsets[storage] = offset + 1
        read = self.dialect.read(_BUFFERS[storage], offset)
        if scalar == "bool":
            return self.dialect.positive(read)
        if scalar != storage:
            return self.dialect.cast(Type(scalar), read)
        return read

    def value(self, kind: Type) -> str:
        if kind.count:
            parts = [self.value(kind.element) for _ in range(kind.count)]
            return self.dialect.array(kind, parts)
        held = tuple_element(kind.opaque)
        if held:
            element, length = held
            parts = [self.element(element.scalar) for _ in range(length)]
            return self.dialect.construct(kind, parts)
        if not kind.shape:
            return self.element(kind.scalar)
        if len(kind.shape) == 1:
            parts = [self.element(kind.scalar) for _ in range(kind.shape[0])]
            return self.dialect.construct(kind, parts)
        columns, rows = kind.shape
        count, size = (rows, columns) if self.dialect.rows_first else (columns, rows)
        inner = Type(kind.scalar, (size,))
        return self.dialect.construct(kind, [self.value(inner) for _ in range(count)])

    @property
    def declarations(self) -> list[str]:
        """The resources the probe declares before its entry point."""
        objects = self.receiver.declarations if self.receiver else []
        return (
            objects
            + self.uniforms.declarations
            + self.shared
            + self.pointers
            + self.handles
        )

    def _resource(self, kind: Type) -> bool:
        if (self.payload or self.tracing) and kind.opaque == "udt":
            return True
        if kind.opaque in _RAY_DESCRIPTIONS or local_object(kind):
            return True
        return (
            bool(self.receiver and self.receiver.takes(kind))
            or self.uniforms.takes(kind)
            or self.dialect.handle(kind, "") is not None
        )

    def _attributes(self, param: Param) -> bool:
        """Whether ``param`` takes hit attributes: a user-defined ``in`` or ``out`` parameter outside a payload stage.
        A user-defined ``inout`` parameter there is a payload, which such a probe does not declare.
        """
        return (
            not self.payload
            and param.type.opaque == "udt"
            and param.direction in ("in", "out")
        )

    def _hit_attributes(self, name: str, filled: bool) -> list[str]:
        """Triangle hit attributes ``name``, with barycentrics the input buffers give when ``filled``."""
        lines = [f"{_HIT_ATTRIBUTES} {name};"]
        if filled:
            lines.append(f"{name}.barycentrics = {self.value(Type('float', (2,)))};")
        return lines

    def _ray(self, name: str) -> list[str]:
        """A ray descriptor ``name`` whose origin and direction the input buffers give."""
        vector = Type("float", (3,))
        return [
            f"RayDesc {name};",
            f"{name}.Origin = {self.value(vector)};",
            f"{name}.TMin = 0.0;",
            f"{name}.Direction = {self.value(vector)};",
            f"{name}.TMax = 1.0;",
        ]

    def _arguments(self, signature: Signature) -> tuple[list[str], Type | None]:
        dialect, arguments, output = self.dialect, [], None
        receiver = self.receiver
        result = signature.result
        read = bool(
            result.opaque
            and (dialect.reading(result.opaque) or dialect.holding(result.opaque))
        )
        params = [p.type for p in signature.params if not self._attributes(p)]
        for kind in ([] if read else [result]) + params:
            if not self._resource(kind):
                # Raises ProbeError for what the language cannot spell.
                dialect.type(kind.element)
        for index, param in enumerate(signature.params):
            local = f"a{index}"
            if self.payload and (
                param.type.opaque == "udt" or param.direction == "inout"
            ):
                arguments.append(dialect.payload)
            elif (
                self.tracing
                and param.type.opaque == "udt"
                and param.direction == "inout"
            ):
                self.traced = True
                arguments.append(dialect.payload)
            elif local_object(param.type):
                self.lines += local_object(param.type, local)
                arguments.append(local)
            elif self._attributes(param):
                self.lines += self._hit_attributes(local, param.direction == "in")
                arguments.append(
                    local if param.direction == "in" else dialect.out_argument(local)
                )
            elif receiver and receiver.takes(param.type):
                arguments.append(receiver.argument(param.type))
            elif self.uniforms.takes(param.type):
                arguments.append(self.uniforms.argument(param.type))
            elif handle := dialect.handle(param.type, local):
                self.lines += handle.lines
                self.handles += handle.declarations
                arguments.append(handle.argument)
            elif param.type.opaque in _RAY_DESCRIPTIONS:
                self.lines += self._ray(local)
                arguments.append(local)
            elif index == 0 and self.interpolant:
                kind = dialect.type(param.type)
                if self.interpolant == "flat":
                    kind = dialect.flat(kind)
                self.inputs.append((kind, local))
                arguments.append(dialect.stage_input(local))
            elif param.direction == "inout":
                if output is not None:
                    raise ProbeError("more than one inout parameter")
                arguments.append(dialect.inout_argument())
                output = param.type
            elif param.direction == "buffer":
                storage = _storage(param.type.scalar)
                if param.type.shape or storage not in _BUFFERS:
                    raise ProbeError(f"no probe buffer holds {param.type.scalar}")
                self.offsets.setdefault(storage, 0)
                arguments.append(dialect.buffer_argument(_BUFFERS[storage]))
            elif param.direction == "out":
                self.lines.append(dialect.declare(param.type, local))
                arguments.append(dialect.out_argument(local))
            elif param.direction == "const" and param.type.count:
                self.lines.append(dialect.constant_array(param.type, local))
                arguments.append(local)
            elif param.direction == "const":
                arguments.append(dialect.constant(param.type))
            elif param.direction == "address":
                value = self.value(param.type)
                self.lines.append(dialect.declare(param.type, local, value))
                arguments.append(dialect.address(local))
            elif param.direction == "workgroup":
                self.shared.append(dialect.workgroup(param.type, local))
                if param.type.count:
                    element = param.type.element
                    self.lines += [
                        f"{local}[{n}] = {self.value(element)};"
                        for n in range(param.type.count)
                    ]
                else:
                    self.lines.append(f"{local} = {self.value(param.type)};")
                arguments.append(dialect.workgroup_argument(local))
            elif param.direction == "pointer":
                self.pointers.append(dialect.pointer(param.type, local))
                arguments.append(local)
            else:
                value = self.value(param.type)
                self.lines.append(dialect.declare(param.type, local, value))
                arguments.append(local)
        return arguments, output

    def body(
        self, name: str, signature: Signature, baseline: bool = False
    ) -> tuple[list[str], Type]:
        """Statements that call ``name`` once, and the element type of the output buffer."""
        dialect = self.dialect
        arguments, output = self._arguments(signature)
        atomic = output is not None
        if self.receiver:
            name = self.receiver.call(name)
        result = signature.result
        assigned = dialect.assignment(name, result, arguments)
        if assigned:
            self.lines += assigned[0]
        call = assigned[1] if assigned else dialect.call(name, arguments)
        reading = dialect.reading(result.opaque) if result.opaque else None
        held = dialect.holding(result.opaque) if result.opaque else ""
        if output is None and reading:
            kind, read = reading
            output, stores = self._stored(kind, read.format(call))
        elif output is None and held:
            output = Type("uint")
            stores = [f"{held} r = {call};", dialect.store(dialect.literal(output))]
        elif output is not None or result.opaque:
            stores = [dialect.statement(call)]
            if output is None:
                output = Type("uint")
                stores.append(dialect.store(dialect.literal(output)))
        elif result.scalar == "bool":
            flag = dialect.first("r") if result.shape else "r"
            stores = [
                dialect.declare(result, "r", call),
                dialect.store(dialect.to_int(flag)),
            ]
            output = Type("int")
        else:
            output, stores = self._stored(result, call)
        if baseline:
            stores = [line for line in [dialect.keep(atomic)] if line]
        return self.lines + stores, output

    def _stored(self, kind: Type, value: str) -> tuple[Type, list[str]]:
        """Statements that keep ``value``, of type ``kind``, in a local and store it, and the output buffer's element
        type: an array's first element, a matrix's first column or row, and a scalar the buffers cannot hold widened
        or narrowed.
        """
        dialect = self.dialect
        output, stored = kind, "r"
        if kind.count:
            output, stored = kind.element, "r[0]"
        if len(output.shape) == 2:
            columns, rows = output.shape
            output = Type(output.scalar, (columns if dialect.rows_first else rows,))
            stored = dialect.part(stored)
        if _storage(kind.scalar) != kind.scalar:
            output = output.with_scalar(_storage(kind.scalar))
            stored = dialect.cast(output, stored)
        return output, [
            dialect.declare(kind, "r", value),
            dialect.store(dialect.stored(output, stored)),
        ]


def build(feature: Feature, language: str, *, baseline: bool = False) -> Probe:
    """The probe for ``feature`` in ``language``, a key of :data:`DIALECTS`: CrossGL (``crossgl``, or for WebGL
    ``crossgl-es``) for the output direction, the feature's own language for input and for lowering to SPIR-V.
    """
    dialect = DIALECTS[language]
    crossgl = dialect.language == "crossgl"
    if constructs.is_construct(feature):
        if baseline:
            raise ProbeError("language-level features have no baseline probe")
        source, entry, suffix = constructs.build(feature, dialect)
        return Probe(language, source, entry, suffix)
    # A counterpart stands in for a signature the probes cannot spell, but only in the CrossGL probe.
    if feature.signature is None and not (crossgl and feature.counterpart):
        raise ProbeError(feature.note or "the feature list gives no signature")
    if feature.stage not in (
        "compute",
        "fragment",
        *_STAGE_PROGRAMS,
        *RAYTRACING_STAGES,
    ):
        raise ProbeError(f"no {feature.stage}-stage probe yet")
    absent = dialect.absent(feature.stage)
    if absent:
        raise NoSpelling(absent)
    name = feature.crossgl if crossgl else feature.spelling or feature.name
    if not name:
        raise NoSpelling(
            "CrossGL has no spelling for this operation"
            + (f": {feature.note}" if feature.note else "")
        )
    # A method is called on an object of its type, which the feature's group names.
    receiver = (
        Receiver(feature.group, dialect, feature.spelling)
        if feature.category == "method"
        else None
    )
    interpolant = feature.name in _INTERPOLANTS and (
        "flat" if feature.name in _FLAT else "smooth"
    )
    builder = _Builder(
        dialect,
        receiver,
        interpolant or "",
        feature.stage in _PAYLOAD_STAGES,
        feature.stage == _TRACING_STAGE,
    )
    signature = feature.signature
    if crossgl and feature.counterpart:
        signature = feature.counterpart
    body, output = builder.body(name, signature, baseline)
    buffers = [
        (dialect.type(Type(scalar)), _BUFFERS[scalar])
        for scalar in ("float", "int", "uint")
        if scalar in builder.offsets
    ]
    atomic = any(p.direction == "inout" for p in signature.params)
    buffers.append((dialect.buffer_element(output, atomic), "outp"))
    declarations = builder.declarations
    if feature.stage == "fragment":
        source = dialect.fragment(
            buffers, body, declarations=declarations, inputs=builder.inputs
        )
        return Probe(
            language, source, dialect.entry, dialect.fragment_suffix or dialect.suffix
        )
    if feature.stage in _STAGE_PROGRAMS:
        program = getattr(dialect, _STAGE_PROGRAMS[feature.stage])
        if receiver and receiver.parameter:
            source = program(
                buffers, body, declarations=declarations, parameter=receiver.parameter
            )
        else:
            source = program(buffers, body, declarations=declarations)
        suffix = dialect.geometry_suffix if feature.stage == "geometry" else ""
        return Probe(language, source, dialect.entry, suffix or dialect.suffix)
    if feature.stage in RAYTRACING_STAGES:
        source = dialect.raytracing(
            feature.stage,
            buffers,
            body,
            declarations=declarations,
            traced=builder.traced,
        )
        return Probe(language, source, dialect.entry, dialect.suffix)
    source = dialect.kernel(buffers, body, declarations=declarations)
    return Probe(language, source, dialect.entry, dialect.suffix)

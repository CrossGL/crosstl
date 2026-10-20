"""Minimal probe programs: one call of one feature inside a compute kernel, or a fragment shader for a feature only
that stage has.

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
from .model import Feature, Signature, Type
from .objects import Receiver

__all__ = ["DIALECTS", "NoSpelling", "Probe", "ProbeError", "build", "spell"]
_BUFFERS = {"float": "fin", "int": "iin", "uint": "uin"}
# Fragment-shader functions whose first argument must be a shader input itself, not a value computed from one.
_INTERPOLANTS = frozenset(
    "interpolateAtCentroid interpolateAtOffset interpolateAtSample InterpolateAtCentroid InterpolateAtOffset "
    "InterpolateAtSample EvaluateAttributeAtCentroid EvaluateAttributeAtSample EvaluateAttributeCentroid "
    "EvaluateAttributeSnapped".split()
)


@dataclass(frozen=True)
class Probe:
    language: str
    source: str
    entry: str
    suffix: str


def spell(kind: Type, language: str) -> str:
    return DIALECTS[language].type(kind)


def _storage(scalar: str) -> str:
    return "float" if scalar in ("bool", "double") else scalar


class _Builder:
    def __init__(
        self,
        dialect: Dialect,
        receiver: Receiver | None = None,
        interpolant: bool = False,
    ):
        self.dialect = dialect
        self.receiver = receiver
        self.interpolant = interpolant
        self.offsets: dict[str, int] = {}
        self.lines: list[str] = list(receiver.locals) if receiver else []
        # Fragment-shader inputs as ``(type spelling, name)``.
        self.inputs: list[Buffer] = []

    def element(self, scalar: str) -> str:
        storage = _storage(scalar)
        offset = self.offsets.get(storage, 0)
        self.offsets[storage] = offset + 1
        read = self.dialect.read(_BUFFERS[storage], offset)
        if scalar == "bool":
            return self.dialect.positive(read)
        if scalar == "double":
            return self.dialect.cast(Type("double"), read)
        return read

    def value(self, kind: Type) -> str:
        if not kind.shape:
            return self.element(kind.scalar)
        if len(kind.shape) == 1:
            parts = [self.element(kind.scalar) for _ in range(kind.shape[0])]
            return self.dialect.construct(kind, parts)
        columns, rows = kind.shape
        count, size = (rows, columns) if self.dialect.rows_first else (columns, rows)
        inner = Type(kind.scalar, (size,))
        return self.dialect.construct(kind, [self.value(inner) for _ in range(count)])

    def _arguments(self, signature: Signature) -> tuple[list[str], Type | None]:
        dialect, arguments, output = self.dialect, [], None
        receiver = self.receiver
        for kind in (signature.result, *(p.type for p in signature.params)):
            if not (receiver and receiver.takes(kind)):
                dialect.type(
                    kind
                )  # raises ProbeError for what the language cannot spell
        for index, param in enumerate(signature.params):
            local = f"a{index}"
            if receiver and receiver.takes(param.type):
                arguments.append(receiver.argument(param.type))
            elif index == 0 and self.interpolant:
                self.inputs.append((dialect.type(param.type), local))
                arguments.append(dialect.stage_input(local))
            elif param.direction == "inout":
                if output is not None:
                    raise ProbeError("more than one inout parameter")
                arguments.append(dialect.inout_argument())
                output = param.type
            elif param.direction == "out":
                self.lines.append(dialect.declare(param.type, local))
                arguments.append(dialect.out_argument(local))
            elif param.direction == "const":
                arguments.append(dialect.literal(param.type))
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
        call = dialect.call(name, arguments)
        result = signature.result
        if output is not None or result.opaque:
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
            output, value = result, "r"
            if len(result.shape) == 2:
                columns, rows = result.shape
                output = Type(result.scalar, (columns if dialect.rows_first else rows,))
                value = dialect.part("r")
            if result.scalar == "double":
                output = output.with_scalar("float")
                value = dialect.cast(output, value)
            stores = [
                dialect.declare(result, "r", call),
                dialect.store(dialect.stored(output, value)),
            ]
        if baseline:
            stores = [line for line in [dialect.keep(atomic)] if line]
        return self.lines + stores, output


def build(feature: Feature, language: str, *, baseline: bool = False) -> Probe:
    """The probe for ``feature`` in ``language``: CrossGL for the output direction, the feature's own language for
    input and for lowering to SPIR-V."""
    if constructs.is_construct(feature):
        if baseline:
            raise ProbeError("language-level features have no baseline probe")
        dialect = DIALECTS[language]
        source, entry, suffix = constructs.build(feature, language, dialect)
        return Probe(language, source, entry, suffix)
    if feature.signature is None:
        raise ProbeError(feature.note or "the feature list gives no signature")
    if feature.stage not in ("compute", "fragment"):
        raise ProbeError(f"no {feature.stage}-stage probe yet")
    name = (
        feature.crossgl if language == "crossgl" else feature.spelling or feature.name
    )
    if not name:
        raise NoSpelling(
            "CrossGL has no spelling for this operation"
            + (f": {feature.note}" if feature.note else "")
        )
    dialect = DIALECTS[language]
    # A method is called on an object of its type, which the feature's group names.
    receiver = (
        Receiver(feature.group, dialect) if feature.category == "method" else None
    )
    builder = _Builder(dialect, receiver, feature.name in _INTERPOLANTS)
    body, output = builder.body(name, feature.signature, baseline)
    buffers = [
        (dialect.type(Type(scalar)), _BUFFERS[scalar])
        for scalar in ("float", "int", "uint")
        if scalar in builder.offsets
    ]
    atomic = any(p.direction == "inout" for p in feature.signature.params)
    buffers.append((dialect.buffer_element(output, atomic), "outp"))
    declarations = receiver.declarations if receiver else []
    if feature.stage == "fragment":
        source = dialect.fragment(
            buffers, body, declarations=declarations, inputs=builder.inputs
        )
        return Probe(
            language, source, dialect.entry, dialect.fragment_suffix or dialect.suffix
        )
    source = dialect.kernel(buffers, body, declarations=declarations)
    return Probe(language, source, dialect.entry, dialect.suffix)

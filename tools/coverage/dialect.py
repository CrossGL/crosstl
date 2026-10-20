"""How one language spells a probe: types, expressions, statements and the kernel around them.

:class:`Dialect` spells everything the way GLSL does; a language overrides only what it writes differently. The probe
builder in :mod:`.probes` talks to a dialect and never to a language directly.
"""

from __future__ import annotations

from .model import Type

# A storage buffer as ``(element type spelling, name)``; buffer ``k`` is bound at binding ``k``.
Buffer = tuple[str, str]


class ProbeError(ValueError):
    """The feature's signature cannot be expressed as a probe yet."""


class NoSpelling(ProbeError):
    """The language has no spelling for the operation, so the operation is a gap rather than an untested probe."""


class Dialect:
    language = ""
    suffix = ""
    entry = "main"
    # The file suffix of a fragment probe, where the language's tools tell the stage from it.
    fragment_suffix = ""
    # HLSL builds matrices from rows and indexes rows; the others use columns.
    rows_first = False
    _PREFIX = {"float": "", "double": "d", "int": "i", "uint": "u", "bool": "b"}

    def type(self, kind: Type) -> str:
        if kind.opaque:
            if kind.opaque == "void":
                return "void"
            raise ProbeError(f"no probe for parameters of type {kind.opaque}")
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

    def literal(self, kind: Type) -> str:
        """A constant of ``kind``, for an argument the language requires to be a constant expression."""
        if kind.shape:
            raise ProbeError("no constant probe argument of a vector or matrix type")
        return {"bool": "false", "int": "0", "uint": "0u"}.get(kind.scalar, "0.0")

    def buffer_element(self, kind: Type, atomic: bool) -> str:
        """The element type of the output buffer; ``atomic`` when the probe passes it to an atomic operation."""
        return self.type(kind)

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

    def call(self, name: str, arguments: list[str]) -> str:
        return f"{name}({', '.join(arguments)})"

    def out_argument(self, name: str) -> str:
        return name

    def inout_argument(self) -> str:
        return "outp[i]"

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
        """
        raise ProbeError(f"{self.language} has no fragment stage")

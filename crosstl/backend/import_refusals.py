"""Typed refusal for source constructs an importer cannot carry into CrossGL.

An importer that meets a source construct CrossGL cannot represent used to write ``/* unsupported ... */`` next to a
neutral value (``true``, ``0.0``, ``vec4(0.0)``) or next to a call that silently dropped arguments such as an LOD
clamp or a tiled-resource status output. The comment disappears as soon as the CrossGL program is parsed, so every
target compiled from it ran with different results. Importers record the reason instead and keep going, so that one
run reports every construct that needs work; :meth:`LoweringRefusals.run` raises the importer's
:class:`ImportRefused` subclass once the program is rendered, and no CrossGL program is returned while a refusal is
pending.
"""

from __future__ import annotations

from crosstl.translator.codegen.lowering_refusals import LoweringRefusals
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "DirectXImportRefused",
    "ImportRefusalMixin",
    "ImportRefused",
    "MetalImportRefused",
    "SlangImportRefused",
]


class ImportRefused(TranslationDiagnostic):
    """One or more source constructs have no faithful CrossGL representation.

    Subclasses set ``source_name`` (the source language key), ``label`` (its name in messages) and ``code``.
    ``reasons`` lists every refusal in emission order, one entry per distinct site.
    """

    source_name = "source"
    label = "The source"
    code = f"{CODE_PREFIX}import-unsupported"

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        self.missing_capabilities = (f"{self.source_name}.import",)
        count = len(self.reasons)
        noun = "construct" if count == 1 else "constructs"
        listed = "; ".join(self.reasons)
        super().__init__(
            f"CrossGL cannot represent {count} {self.label} {noun}: {listed}",
            target="cgl",
        )


class DirectXImportRefused(ImportRefused):
    """One or more HLSL constructs have no faithful CrossGL representation."""

    source_name = "directx"
    label = "HLSL"
    code = f"{CODE_PREFIX}directx-import-unsupported"


class MetalImportRefused(ImportRefused):
    """One or more Metal Shading Language constructs have no faithful CrossGL representation."""

    source_name = "metal"
    label = "Metal"
    code = f"{CODE_PREFIX}metal-import-unsupported"


class SlangImportRefused(ImportRefused):
    """One or more Slang constructs have no faithful CrossGL representation."""

    source_name = "slang"
    label = "Slang"
    code = f"{CODE_PREFIX}slang-import-unsupported"


class ImportRefusalMixin:
    """``refuse_import`` for an importer whose generation runs through ``self.refusals.run``.

    Subclasses set ``IMPORT_REFUSAL_TYPE`` to their :class:`ImportRefused` subclass. The sink is created on first
    use, so importers built without ``__init__`` (as some tests do) still refuse.
    """

    IMPORT_REFUSAL_TYPE = ImportRefused

    @property
    def refusals(self):
        sink = self.__dict__.get("_import_refusals")
        if sink is None:
            sink = LoweringRefusals(self.IMPORT_REFUSAL_TYPE)
            self.__dict__["_import_refusals"] = sink
        return sink

    def refuse_import(self, reason, placeholder=""):
        """Record a construct with no faithful CrossGL representation and return ``placeholder`` in its place.

        The placeholder only lets generation reach the other refusals in the program; no program is returned while
        a refusal is pending.
        """
        self.refusals.record(reason)
        return placeholder

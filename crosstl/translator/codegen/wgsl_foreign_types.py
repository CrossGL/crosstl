"""Built-in types of other languages that WGSL output would still name, refused.

The importers carry the built-in types of the languages they read into the IR as plain names
(:data:`~crosstl.translator.semantics.builtin_type_names.BUILTIN_TYPE_NAMES`). The WGSL generator spells the ones it
knows (scalars, vectors, matrices, textures, samplers and buffers) and writes any other type name as the name of a
declaration of the program. A built-in name it has no spelling for therefore came out unchanged, as
``tex: cudaTextureObject_t`` or ``var vft: visible_function_table``, which no WGSL compiler accepts ("no definition in
scope for identifier").

As in :mod:`.written_type_names`, the refusal is decided on the output, because the generator also maps names of
declarations that a later lowering replaces: :meth:`WGSLForeignTypeMixin.record_foreign_wgsl_type` records a built-in
name the program does not declare when it is mapped, and :meth:`WGSLForeignTypeMixin.refuse_written_foreign_wgsl_types`
refuses each recorded name the finished module still uses as an identifier, with :class:`WGSLLoweringRefused`.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX
from ..semantics.builtin_type_names import BUILTIN_TYPE_NAMES
from .lowering_refusals import LoweringRefused
from .written_type_names import written_foreign_type_names

__all__ = ["WGSLForeignTypeMixin", "WGSLLoweringRefused"]

# Built-in names that are WGSL's own predeclared types (WGSL section 6, "Types").
_WGSL_TYPE_NAME = re.compile(
    r"(?:bool|f16|f32|i32|u32|sampler|sampler_comparison|vec[234][hfiu]?|mat[234]x[234][hf]?)"
)


class WGSLLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful WGSL lowering."""

    target_name = "wgsl"
    label = "WGSL"
    code = f"{CODE_PREFIX}wgsl-lowering-unsupported"


class WGSLForeignTypeMixin:
    """Records and refuses foreign built-in type names in WGSL output."""

    def record_foreign_wgsl_type(self, source_name, written_name):
        if source_name not in BUILTIN_TYPE_NAMES or _WGSL_TYPE_NAME.fullmatch(
            written_name
        ):
            return
        if getattr(self, "written_foreign_wgsl_types", None) is None:
            self.written_foreign_wgsl_types = {}
        self.written_foreign_wgsl_types[written_name] = source_name

    def refuse_written_foreign_wgsl_types(self, code):
        recorded = getattr(self, "written_foreign_wgsl_types", None) or {}
        written = written_foreign_type_names(code, set(recorded))
        if written:
            raise WGSLLoweringRefused(
                [
                    f"type: the built-in type '{recorded[name]}' has no WGSL spelling"
                    for name in written
                ]
            )
        return code

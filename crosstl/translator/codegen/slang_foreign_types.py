"""Built-in types of other languages that Slang output would still name, refused.

The importers carry the built-in types of the languages they read into the IR as plain names
(:data:`~crosstl.translator.semantics.builtin_type_names.BUILTIN_TYPE_NAMES`). The Slang generator spells the ones it
knows and writes any other type name unchanged, as the name of a declaration of the program. A built-in name of
another language therefore came out as written, as ``void probe(cudaTextureObject_t tex)``, which slangc rejects
("undefined identifier 'cudaTextureObject_t'", E30015).

As in :mod:`.written_type_names`, the refusal is decided on the output, because the generator also maps names it
never writes: :func:`record_slang_type_name` records an unmapped built-in name that neither Slang nor the program
declares, and :func:`refuse_written_slang_foreign_types` refuses each recorded name the finished module still uses.
Slang's own type names are HLSL's (Slang accepts every HLSL type) and the Slang core module's types in
:data:`SLANG_OWN_TYPE_NAMES` (``hlsl.meta.slang`` and ``core.meta.slang``, Slang v2026.20).
"""

from __future__ import annotations

from ..semantics.builtin_type_names import BUILTIN_TYPE_NAMES
from .written_type_names import HLSL_TYPE_NAMES, written_foreign_type_names

__all__ = [
    "SLANG_OWN_TYPE_NAMES",
    "record_slang_type_name",
    "refuse_written_slang_foreign_types",
]

# Built-in type names of other languages that the Slang core module declares too.
SLANG_OWN_TYPE_NAMES = frozenset(
    {
        "HitObject",
        "RWTextureCube",
        "RWTextureCubeArray",
        "RasterizerOrderedTexture2DMS",
        "RasterizerOrderedTexture2DMSArray",
        "RasterizerOrderedTextureCube",
        "RasterizerOrderedTextureCubeArray",
        "Self",
        "int8_t",
        "uint8_t",
    }
)


def record_slang_type_name(codegen, name):
    """``name`` unchanged, recorded when it is a foreign built-in type the rendering in progress writes."""
    foreign = getattr(codegen, "slang_foreign_type_names", None)
    if (
        foreign is not None
        and name in BUILTIN_TYPE_NAMES
        and name not in HLSL_TYPE_NAMES
        and name not in SLANG_OWN_TYPE_NAMES
        and name not in codegen.user_symbol_names
    ):
        foreign.add(name)
    return name


def refuse_written_slang_foreign_types(codegen, code):
    """Refuse each recorded foreign built-in type that the rendered Slang ``code`` still names."""
    for name in written_foreign_type_names(
        code, getattr(codegen, "slang_foreign_type_names", None) or set()
    ):
        codegen.refuse_lowering(
            f"type: the built-in type '{name}' of another language has no Slang spelling",
            name,
        )

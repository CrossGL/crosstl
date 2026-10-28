"""Metal's ``[[invariant]]`` attribute for an invariant vertex position.

MSL 2.1 and later (Metal Shading Language Specification section 5.2.3.3) mark a vertex output invariant with
``[[invariant]]``, and only on the member that carries ``[[position]]``. A CrossGL output declared
``vec4 position @ gl_Position @ invariant`` used to lose the qualifier. Invariance of any other output has no Metal
spelling, so it is refused rather than dropped. The attribute takes effect when the host compiles the library
with ``MTLCompileOptions.preserveInvariance`` (``-fpreserve-invariance`` for the command-line compiler).
"""

from __future__ import annotations

from .interface_qualifiers import has_invariant_attribute

__all__ = ["metal_invariant_attribute_suffix"]


def metal_invariant_attribute_suffix(codegen, node, semantic=None) -> str:
    """`` [[invariant]]`` for an invariant position output, ``""`` for a value without the qualifier.

    ``semantic`` defaults to the semantic of ``node``; an entry function passes its return semantic.
    """
    if not has_invariant_attribute(node):
        return ""
    if semantic is None:
        semantic = codegen.semantic_from_node(node)
    if (
        semantic is not None
        and codegen.canonical_metal_semantic(semantic) == "position"
    ):
        return " [[invariant]]"
    from .metal_codegen import UnsupportedMetalFeatureError

    name = getattr(node, "name", "<value>")
    raise UnsupportedMetalFeatureError(
        "invariant-output",
        f"Metal codegen cannot make '{name}' invariant: MSL's [[invariant]] attribute applies only to the "
        "vertex position output (Metal Shading Language Specification section 5.2.3.3)",
        missing_capabilities=("metal.invariant-output",),
        operation=str(name),
        reason="invariant-non-position-output",
        source_location=getattr(node, "source_location", None),
    )

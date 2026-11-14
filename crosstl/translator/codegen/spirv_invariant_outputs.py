"""``Invariant`` decorations on SPIR-V built-in outputs.

An entry-point output that carries a built-in semantic becomes the built-in variable itself, which the generator
creates once and shares between every value naming it. Its ``@invariant`` qualifier used to be dropped on that
path, so ``vec4 position @ gl_Position @ invariant`` lost the ``Invariant`` decoration (SPIR-V 1.6 section 3.20)
that ``invariant gl_Position;`` gets from glslang. :func:`decorate_invariant_builtin_output` adds it once per
variable.
"""

from __future__ import annotations

from .interface_qualifiers import has_invariant_attribute

__all__ = ["decorate_invariant_builtin_output"]


def decorate_invariant_builtin_output(codegen, variable, node, storage_class) -> None:
    """Decorate the built-in output ``variable`` ``Invariant`` when ``node`` declares it invariant."""
    if storage_class != "Output" or node is None or variable is None:
        return
    if not has_invariant_attribute(node):
        return
    decoration = f"OpDecorate %{variable.id} Invariant"
    if decoration not in codegen.decorations:
        codegen.decorations.append(decoration)

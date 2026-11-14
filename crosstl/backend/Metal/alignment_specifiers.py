"""The alignment a Metal ``alignas`` specifier list gives a struct.

MSL is based on C++14 (MSL 3.2 section 1.4), where ``alignas(N)`` and ``alignas(T)`` on a class raise its alignment
to the strictest specifier, ``alignas(T)`` meaning ``alignas(alignof(T))``, and an ``alignas(0)`` is ignored
(C++14 [dcl.align] paragraphs 2, 3 and 4). A struct's size is a multiple of its alignment, so ``struct alignas(16)
R { float a; float b; }`` takes 16 bytes where its members take 8, and a ``device R*`` buffer has a 16-byte stride.

:meth:`MetalAlignmentSpecifiers.metal_alignment_specifier_value` folds a parsed specifier list into that alignment.
The importer spells it ``@metal_alignas(N)`` on the CrossGL struct, which the Metal generator declares again and
the layout rules for ``@buffer_layout(metal)`` buffers apply, so the other targets read the elements at their MSL
offsets.
"""

from __future__ import annotations

from ...translator.codegen.array_utils import evaluate_literal_int_expression

__all__ = ["MetalAlignmentSpecifiers"]


class MetalAlignmentSpecifiers:
    """Mixin for :class:`MetalToCrossGLConverter`: the value of an ``alignas`` specifier list."""

    def metal_alignment_specifier_value(self, specifiers):
        """The strictest alignment of ``specifiers``, or ``None`` when one of them does not fold to a power of two.

        Each specifier is an integral constant expression (a string or an expression node) or ``("type", T)``.
        ``0`` for a list of only ``alignas(0)`` specifiers, which leave the natural alignment.
        """
        alignment = 0
        for specifier in specifiers or ():
            value = self.metal_alignment_specifier_part(specifier)
            if value is None or value < 0 or value & (value - 1):
                return None
            alignment = max(alignment, value)
        return alignment

    def metal_struct_alignment_attributes(self, struct_node, pad="    "):
        """The ``@metal_alignas`` lines that precede the CrossGL declaration of ``struct_node``.

        A list that folds yields one ``@metal_alignas(N)``, and none when it only holds ``alignas(0)``. A list that
        does not fold (a dependent ``alignas(T)`` of a generic struct) keeps one attribute per specifier, which the
        generators and layout rules refuse rather than lay the struct out at its natural alignment.
        """
        specifiers = list(getattr(struct_node, "alignas", None) or ())
        if not specifiers:
            return ""
        alignment = self.metal_alignment_specifier_value(specifiers)
        if alignment is not None:
            return f"{pad}@metal_alignas({alignment})\n" if alignment else ""
        lines = []
        for specifier in specifiers:
            if isinstance(specifier, tuple) and specifier[0] == "type":
                text = self.map_type(specifier[1])
            else:
                text = self.generate_expression(specifier, False)
            lines.append(f"{pad}@metal_alignas({text})\n")
        return "".join(lines)

    def metal_alignment_specifier_part(self, specifier):
        if isinstance(specifier, tuple):
            if len(specifier) != 2 or specifier[0] != "type":
                return None
            layout = self.metal_concrete_type_layout(str(specifier[1]))
            return None if layout is None else layout[1]
        if isinstance(specifier, str):
            value = self.evaluate_concrete_array_extent(specifier)
        else:
            value = evaluate_literal_int_expression(specifier)
            if value is None:
                # ``sizeof`` and named constants fold only once rendered.
                value = self.evaluate_concrete_array_extent(
                    self.format_array_extent(specifier)
                )
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

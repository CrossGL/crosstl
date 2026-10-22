"""GLSL arrays of uniform blocks, imported as CrossGL arrays of constant buffers.

``layout(binding = 4) uniform Weights { vec4 data[64]; } weights[4];`` declares four uniform blocks, each bound to
a buffer of its own (GLSL 4.60 section 4.3.9: "an array of blocks ... each individual array element corresponds
to a separate buffer object bound to a binding point"); an unsized ``weights[]`` is a runtime-sized descriptor
array (GL_EXT_nonuniform_qualifier). CrossGL spells a set of buffers as an array of ``ConstantBuffer<T>``, so the
import is ``ConstantBuffer<Weights> weights[4] @binding(4);``. A ``uniform Weights weights[4]`` declaration would
instead mean one buffer holding four ``Weights`` values, which binds and lays out differently on every target.

A CrossGL constant buffer is laid out with std140, the default layout of a GLSL uniform block (section 4.4.5). An
array of blocks declared ``std430`` or ``scalar`` has no CrossGL spelling yet and is refused with
:class:`UniformBlockArrayLayoutUnsupportedError` rather than re-laid-out silently.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "UniformBlockArrayLayoutUnsupportedError",
    "arrayed_uniform_block_declaration",
]


class UniformBlockArrayLayoutUnsupportedError(TranslationDiagnostic):
    """An array of uniform blocks uses a memory layout CrossGL constant buffers cannot carry."""

    def __init__(self, block_name, instance_name, layout):
        self.missing_capabilities = ("crossgl.constant-buffer-array-layout",)
        super().__init__(
            f"array of uniform blocks '{block_name} {instance_name}[]' is declared {layout}: CrossGL arrays of "
            "constant buffers are laid out std140 (GLSL 4.60 section 4.4.5), and no other layout is expressible",
            target="crossgl",
            code=f"{CODE_PREFIX}uniform-block-array-layout-unsupported",
        )


def arrayed_uniform_block_declaration(codegen, var):
    """The CrossGL ``ConstantBuffer<T> name[N] @set(..) @binding(..);`` line of a GLSL array of uniform blocks."""
    block_name = codegen.uniform_block_name(var)
    layout = codegen.uniform_block_layout(block_name, [var])
    for name in ("std430", "scalar"):
        if codegen.layout_has_key(layout, name):
            raise UniformBlockArrayLayoutUnsupportedError(block_name, var.name, name)
    attributes = codegen.descriptor_set_uniform_block_attribute_suffix(
        block_name, [var]
    )
    element_type = codegen.convert_type(block_name)
    array_suffix = codegen.array_suffix(var)
    return f"ConstantBuffer<{element_type}> {var.name}{array_suffix}{attributes};\n"
